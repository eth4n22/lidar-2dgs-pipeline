#!/usr/bin/env python3
"""
2DGS Converter GUI

A graphical interface for converting LiDAR point clouds to 2DGS surfels.

Usage:
    python gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import PIL for image handling
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Note: PIL not installed. Logo will not be displayed.")
    print("Install with: pip install Pillow")

from src.txt_io import load_xyzrgb_txt, validate_format
from src.preprocess import normalize_points, voxel_downsample, remove_outliers
from src.normals import estimate_normals_knn, orient_normals_consistently, refine_normals
from src.surfels import build_surfels
from src.export_ply import write_ply


# Color scheme
COLORS = {
    'bg': '#0F172A',           # Dark background
    'card_bg': '#111827',      # Card background
    'border': '#1F2937',       # Border color
    'accent': '#3B82F6',       # Blue accent
    'accent_hover': '#2563EB',
    'success': '#22C55E',      # Green success
    'warning': '#F59E0B',      # Yellow warning
    'error': '#EF4444',        # Red error
    'text': '#F9FAFB',         # White text
    'text_secondary': '#9CA3AF',  # Gray text
    'input_bg': '#1F2937',     # Input background
}


class LogCapture:
    """Capture print output to GUI."""
    def __init__(self, text_widget, callback=None):
        self.text_widget = text_widget
        self.callback = callback
        
    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()
        if self.callback:
            self.callback(message)
            
    def flush(self):
        pass


class StyledButton(tk.Canvas):
    """Custom styled button canvas."""
    def __init__(self, parent, text="", command=None, primary=False, **kwargs):
        super().__init__(parent, **kwargs)
        self.command = command
        self.primary = primary
        self.hovered = False
        self._text = text
        
        # Configure canvas
        self.configure(width=120, height=36, bg=COLORS['card_bg'], 
                      highlightthickness=0, cursor='hand2')
        
        # Draw button
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self._draw()
        
    def _draw(self):
        self.delete("all")
        
        # Button colors
        bg_color = COLORS['accent'] if self.primary else COLORS['border']
        if self.hovered:
            bg_color = COLORS['accent_hover'] if self.primary else '#374151'
            
        # Draw rounded rectangle
        self._create_rounded_rect(2, 2, self.winfo_reqwidth()-2, self.winfo_reqheight()-2, 
                                  radius=8, fill=bg_color, outline=bg_color)
        
        # Draw text
        text_color = '#FFFFFF' if self.primary else COLORS['text']
        self.create_text(self.winfo_reqwidth()/2, self.winfo_reqheight()/2,
                        text=self._text, fill=text_color, font=('Segoe UI', 10, 'bold'))
        
    def _create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Create rounded rectangle."""
        self.create_arc(x1, y1, x1+2*radius, y1+2*radius, start=90, extent=90, **kwargs)
        self.create_arc(x2-2*radius, y1, x2, y1+2*radius, start=0, extent=90, **kwargs)
        self.create_arc(x1, y2-2*radius, x1+2*radius, y2, start=180, extent=90, **kwargs)
        self.create_arc(x2-2*radius, y2-2*radius, x2, y2, start=270, extent=90, **kwargs)
        self.create_rectangle(x1+radius, y1, x2-radius, y2, **kwargs)
        self.create_rectangle(x1, y1+radius, x2, y2-radius, **kwargs)
        
    def _on_enter(self, event):
        self.hovered = True
        self._draw()
        
    def _on_leave(self, event):
        self.hovered = False
        self._draw()
        
    def _on_click(self, event):
        if self.command:
            self.command()
            
    def set_text(self, text):
        self._text = text
        self._draw()


class SegmentedToggle(tk.Frame):
    """Segmented toggle button for ASCII/Binary selection."""
    def __init__(self, parent, options=None, command=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.options = options or ["Option 1", "Option 2"]
        self.command = command
        self.selected = 0
        self.buttons = []
        
        self.configure(bg=COLORS['card_bg'])
        
        for i, opt in enumerate(self.options):
            btn = tk.Frame(self, bg=COLORS['border'], cursor='hand2')
            btn.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 0))
            btn.bind("<Button-1>", lambda e, idx=i: self._select(idx))
            
            label = tk.Label(btn, text=opt, bg=COLORS['border'], 
                           fg=COLORS['text_secondary'], font=('Segoe UI', 9),
                           padx=12, pady=6)
            label.pack()
            label.bind("<Button-1>", lambda e, idx=i: self._select(idx))
            
            self.buttons.append(label)
            
        self._update_colors()
        
    def _select(self, index):
        if self.selected != index:
            self.selected = index
            self._update_colors()
            if self.command:
                self.command(self.options[index])
                
    def _update_colors(self):
        for i, btn in enumerate(self.buttons):
            if i == self.selected:
                btn.configure(bg=COLORS['accent'], fg='#FFFFFF')
            else:
                btn.configure(bg=COLORS['border'], fg=COLORS['text_secondary'])
                
    def get_selected(self):
        return self.options[self.selected]


class ConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LiDAR to 2DGS Converter")
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)
        self.root.configure(bg=COLORS['bg'])
        
        # Project paths
        self.project_dir = Path(__file__).parent
        self.input_dir = self.project_dir / "data" / "input"
        self.output_dir = self.project_dir / "data" / "output"
        
        # State
        self.selected_input_file = None
        self.selected_output_file = None
        self.processing = False
        self.current_stage = ""
        self.progress_value = 0
        
        # Setup styles
        self.setup_styles()
        
        # Setup UI
        self.setup_ui()
        self.refresh_input_list()
        self.refresh_output_list()
        
    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure frame styles
        style.configure('Card.TFrame', background=COLORS['card_bg'])
        style.configure('TFrame', background=COLORS['bg'])
        style.configure('TLabel', background=COLORS['card_bg'], foreground=COLORS['text'])
        style.configure('TLabelframe', background=COLORS['card_bg'], 
                       foreground=COLORS['text'])
        style.configure('TLabelframe.Label', background=COLORS['card_bg'], 
                       foreground=COLORS['text'], font=('Segoe UI', 10, 'bold'))
        
        # Configure entry styles
        style.configure('TEntry', fieldbackground=COLORS['input_bg'], 
                       foreground=COLORS['text'], borderwidth=0)
        style.map('TEntry', fieldbackground=[('focus', COLORS['accent'])])
        
        # Configure progress bar
        style.configure('Horizontal.TProgressbar', 
                       troughcolor=COLORS['border'],
                       background=COLORS['accent'],
                       thickness=6)
        
        # Configure scrollbar
        style.configure('Vertical.TScrollbar', 
                       troughcolor=COLORS['bg'],
                       background=COLORS['border'],
                       arrowcolor=COLORS['text'])
        
    def setup_ui(self):
        """Create the GUI layout."""
        # ==== HEADER ====
        header = tk.Frame(self.root, bg=COLORS['card_bg'], height=120)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # Logo
        logo_path = self.project_dir / "HDB4.png"
        if HAS_PIL and logo_path.exists():
            try:
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((100, 100), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                
                logo_label = tk.Label(header, image=self.logo_photo, bg=COLORS['card_bg'])
                logo_label.image = self.logo_photo
                logo_label.pack(side=tk.LEFT, padx=16, pady=10)
            except Exception as e:
                print(f"Warning: Could not load logo: {e}")
        
        # Title
        title_label = tk.Label(header, text="LiDAR to 2DGS Converter", 
                              bg=COLORS['card_bg'], fg=COLORS['text'],
                              font=('Segoe UI', 14, 'bold'))
        title_label.pack(side=tk.LEFT, pady=10)
        
        # Status label
        self.status_label = tk.Label(header, text="Idle", 
                                    bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                                    font=('Segoe UI', 10))
        self.status_label.pack(side=tk.RIGHT, padx=16, pady=10)
        
        # Status indicator
        self.status_indicator = tk.Canvas(header, width=10, height=10, 
                                         bg=COLORS['card_bg'], highlightthickness=0)
        self.status_indicator.pack(side=tk.RIGHT, pady=18)
        self.status_indicator.create_oval(0, 0, 10, 10, fill=COLORS['text_secondary'], outline='')
        
        # ==== MAIN CONTENT ====
        main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(12, 16))
        
        # Two-column layout
        left_column = tk.Frame(main_frame, bg=COLORS['bg'], width=400)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_column.pack_propagate(False)
        
        right_column = tk.Frame(main_frame, bg=COLORS['bg'], width=300)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_column.pack_propagate(False)
        
        # ==== LEFT COLUMN ====
        
        # Input Files Card
        self.create_card(left_column, "Input Files")
        input_frame = self.current_card
        
        # File list
        list_container = tk.Frame(input_frame, bg=COLORS['card_bg'], height=100)
        list_container.pack(fill=tk.X, pady=(8, 8))
        list_container.pack_propagate(False)
        
        self.input_listbox = tk.Listbox(list_container, selectmode=tk.SINGLE,
                                        font=('Consolas', 10),
                                        bg=COLORS['input_bg'], fg=COLORS['text'],
                                        bd=0, highlightthickness=1,
                                        highlightbackground=COLORS['border'],
                                        selectbackground=COLORS['accent'],
                                        selectforeground='#FFFFFF')
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL,
                                  command=self.input_listbox.yview)
        self.input_listbox.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self.input_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.input_listbox.bind('<<ListboxSelect>>', self.on_input_selected)
        
        # Buttons
        btn_row = tk.Frame(input_frame, bg=COLORS['card_bg'])
        btn_row.pack(fill=tk.X, pady=(0, 8))
        
        self.refresh_input_btn = self.create_header_button(btn_row, "↻ Refresh", 
                                                          self.refresh_input_list)
        self.refresh_input_btn.pack(side=tk.LEFT, padx=8)
        
        self.add_file_btn = self.create_header_button(btn_row, "+ Add File...", 
                                                      self.add_input_file)
        self.add_file_btn.pack(side=tk.RIGHT, padx=8)
        
        # Processing Options Card
        self.create_card(left_column, "Processing Options")
        options_frame = self.current_card
        
        # Output format toggle
        format_label = tk.Label(options_frame, text="Output Format:", 
                               bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                               font=('Segoe UI', 9))
        format_label.pack(anchor=tk.W, pady=(8, 4), padx=12)
        
        self.format_toggle = SegmentedToggle(options_frame, 
                                            options=["ASCII", "Binary"],
                                            command=self.on_format_changed)
        self.format_toggle.pack(anchor=tk.W, padx=8, pady=(0, 12))
        
        # Parameters
        params = [
            ("Voxel Size (m):", "", "Leave empty to disable downsampling"),
            ("KNN Normals:", "30", "Number of neighbors for normal estimation"),
            ("Refine Iterations:", "0", "Normal refinement iterations"),
        ]
        
        self.param_entries = {}
        for label_text, default, tooltip in params:
            row = tk.Frame(options_frame, bg=COLORS['card_bg'])
            row.pack(fill=tk.X, padx=12, pady=2)
            
            lbl = tk.Label(row, text=label_text, 
                          bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                          font=('Segoe UI', 9), width=18, anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            
            entry = tk.Entry(row, bg=COLORS['input_bg'], fg=COLORS['text'],
                           font=('Consolas', 10), bd=0, highlightthickness=1,
                           highlightbackground=COLORS['border'],
                           highlightcolor=COLORS['accent'],
                           width=12)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT)
            self.param_entries[label_text] = entry
        
        # Checkboxes
        checkbox_frame = tk.Frame(options_frame, bg=COLORS['card_bg'])
        checkbox_frame.pack(fill=tk.X, padx=8, pady=8)
        
        self.outliers_var = tk.BooleanVar()
        self.outliers_cb = tk.Checkbutton(checkbox_frame, text="Remove Outliers",
                                         variable=self.outliers_var,
                                         bg=COLORS['card_bg'], fg=COLORS['text'],
                                         selectcolor=COLORS['card_bg'],
                                         activebackground=COLORS['card_bg'],
                                         font=('Segoe UI', 9),
                                         cursor='hand2')
        self.outliers_cb.pack(anchor=tk.W, padx=4)
        
        # Run Card
        self.create_card(left_column, "Run")
        run_frame = self.current_card
        
        # Progress bar
        progress_label = tk.Label(run_frame, text="Progress", 
                                 bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                                 font=('Segoe UI', 9))
        progress_label.pack(anchor=tk.W, padx=12, pady=(8, 4))
        
        self.progress_bar = ttk.Progressbar(run_frame, mode='determinate', 
                                           style='Horizontal.TProgressbar')
        self.progress_bar.pack(fill=tk.X, padx=12, pady=(0, 4))
        
        self.progress_label = tk.Label(run_frame, text="0%", 
                                      bg=COLORS['card_bg'], fg=COLORS['text'],
                                      font=('Segoe UI', 9))
        self.progress_label.pack(anchor=tk.E, padx=12)
        
        # Stage label
        self.stage_label = tk.Label(run_frame, text="Ready to process", 
                                   bg=COLORS['card_bg'], fg=COLORS['accent'],
                                   font=('Segoe UI', 9))
        self.stage_label.pack(anchor=tk.W, padx=12, pady=(4, 8))
        
        # Start button
        self.start_btn = tk.Button(run_frame, text="START PROCESSING", 
                                    command=self.start_conversion,
                                    bg=COLORS['accent'], fg='#FFFFFF',
                                    font=('Segoe UI', 11, 'bold'),
                                    bd=0, padx=20, pady=8,
                                    cursor='hand2', activebackground=COLORS['accent_hover'])
        self.start_btn.pack(pady=(8, 8))
        
        # Status
        self.process_status = tk.Label(run_frame, text="No file selected", 
                                      bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                                      font=('Segoe UI', 9))
        self.process_status.pack(pady=(0, 8))
        
        # Logs Card
        self.create_card(left_column, "Logs")
        log_frame = self.current_card
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10,
                                                  font=('Consolas', 9),
                                                  bg=COLORS['input_bg'], 
                                                  fg=COLORS['text'],
                                                  bd=0, highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # ==== RIGHT COLUMN ====
        
        # Viewer Card
        self.create_card(right_column, "Viewer")
        viewer_frame = self.current_card
        
        # File list
        list_container = tk.Frame(viewer_frame, bg=COLORS['card_bg'], height=80)
        list_container.pack(fill=tk.X, pady=(8, 8))
        list_container.pack_propagate(False)
        
        self.output_listbox = tk.Listbox(list_container, selectmode=tk.SINGLE,
                                         font=('Consolas', 10),
                                         bg=COLORS['input_bg'], fg=COLORS['text'],
                                         bd=0, highlightthickness=1,
                                         highlightbackground=COLORS['border'],
                                         selectbackground=COLORS['accent'],
                                         selectforeground='#FFFFFF')
        scrollbar2 = ttk.Scrollbar(list_container, orient=tk.VERTICAL,
                                   command=self.output_listbox.yview)
        self.output_listbox.config(yscrollcommand=scrollbar2.set)
        
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self.output_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.output_listbox.bind('<<ListboxSelect>>', self.on_output_selected)
        
        # Buttons
        btn_row = tk.Frame(viewer_frame, bg=COLORS['card_bg'])
        btn_row.pack(fill=tk.X, pady=(0, 8))
        
        self.refresh_output_btn = self.create_header_button(btn_row, "Refresh",
                                                            self.refresh_output_list)
        self.refresh_output_btn.pack(side=tk.LEFT, padx=8)
        
        # View button
        self.view_btn = ttk.Button(btn_row, text="View", 
                                   command=self.view_selected,
                                   state=tk.DISABLED)
        self.view_btn.pack(side=tk.RIGHT, padx=8)
        
        # Delete button  
        self.delete_btn = ttk.Button(btn_row, text="Delete",
                                       command=self.delete_selected,
                                       state=tk.DISABLED)
        self.delete_btn.pack(side=tk.RIGHT, padx=4)
        
    def create_card(self, parent, title):
        """Create a card container."""
        card = tk.Frame(parent, bg=COLORS['card_bg'], bd=1, relief=tk.SOLID,
                       highlightbackground=COLORS['border'], highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 12))
        self.current_card = card
        
        # Card title
        title_label = tk.Label(card, text=title, bg=COLORS['card_bg'], 
                              fg=COLORS['text'], font=('Segoe UI', 10, 'bold'),
                              padx=12, pady=8)
        title_label.pack(anchor=tk.W)
        
        return card
        
    def create_header_button(self, parent, text, command):
        """Create a header-style button."""
        btn = tk.Label(parent, text=text, bg=COLORS['card_bg'],
                      fg=COLORS['accent'], font=('Segoe UI', 9),
                      cursor='hand2', padx=8, pady=4)
        btn.bind("<Button-1>", lambda e: command())
        btn.bind("<Enter>", lambda e: btn.configure(fg=COLORS['accent_hover']))
        btn.bind("<Leave>", lambda e: btn.configure(fg=COLORS['accent']))
        return btn
        
    def on_format_changed(self, value):
        """Handle format toggle change."""
        pass  # Already handled by binary_var tracking
        
    def set_status(self, status, color=None):
        """Update status indicator and label."""
        self.status_label.configure(text=status)
        
        # Update indicator color
        self.status_indicator.delete("all")
        
        status_colors = {
            'Idle': COLORS['text_secondary'],
            'Processing...': COLORS['warning'],
            'Done': COLORS['success'],
            'Error': COLORS['error'],
        }
        
        fill_color = color or status_colors.get(status, COLORS['text_secondary'])
        self.status_indicator.create_oval(0, 0, 10, 10, fill=fill_color, outline='')
        
    def set_progress(self, value, stage=""):
        """Update progress bar and stage label."""
        self.progress_bar['value'] = value
        self.progress_label['text'] = f"{value}%"
        if stage:
            self.stage_label['text'] = stage
        self.root.update_idletasks()
        
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format (uses decimal MB)."""
        if size_bytes >= 1_000_000_000:
            return f"{size_bytes / 1_000_000_000:.2f} GB"
        elif size_bytes >= 1_000_000:
            return f"{size_bytes / 1_000_000:.2f} MB"
        elif size_bytes >= 1_000:
            return f"{size_bytes / 1_000:.2f} KB"
        else:
            return f"{size_bytes} B"

    def refresh_input_list(self):
        """Refresh the list of input TXT files."""
        self.input_listbox.delete(0, tk.END)

        if not self.input_dir.exists():
            self.input_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(self.input_dir.glob("*.txt"))
        for f in txt_files:
            size_str = self.format_file_size(f.stat().st_size)
            self.input_listbox.insert(tk.END, f"{f.name} ({size_str})")
            
        if not txt_files:
            self.input_listbox.insert(tk.END, "(No .txt files found)")
            
    def refresh_output_list(self):
        """Refresh the list of output PLY files."""
        self.output_listbox.delete(0, tk.END)

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        ply_files = sorted(self.output_dir.glob("*.ply"))
        for f in ply_files:
            size_str = self.format_file_size(f.stat().st_size)
            self.output_listbox.insert(tk.END, f"{f.name} ({size_str})")
            
        if not ply_files:
            self.output_listbox.insert(tk.END, "(No .ply files found)")
            
    def add_input_file(self):
        """Open file dialog to add a new input file."""
        filename = filedialog.askopenfilename(
            title="Select TXT Point Cloud",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            src = Path(filename)
            dst = self.input_dir / src.name
            import shutil
            shutil.copy2(src, dst)
            self.refresh_input_list()
            
    def on_input_selected(self, event):
        """Handle input file selection."""
        selection = self.input_listbox.curselection()
        if selection:
            text = self.input_listbox.get(selection[0])
            filename = text.rsplit(" (", 1)[0]
            self.selected_input_file = self.input_dir / filename
            self.process_status.configure(
                text=f"Selected: {filename}",
                fg=COLORS['text']
            )
        else:
            self.selected_input_file = None
            self.process_status.configure(
                text="No file selected",
                fg=COLORS['text_secondary']
            )
            
    def on_output_selected(self, event):
        """Handle output file selection."""
        selection = self.output_listbox.curselection()
        if selection:
            text = self.output_listbox.get(selection[0])
            filename = text.rsplit(" (", 1)[0]
            self.selected_output_file = self.output_dir / filename
            self.view_btn.configure(state=tk.NORMAL)
            self.delete_btn.configure(state=tk.NORMAL)
        else:
            self.selected_output_file = None
            self.view_btn.configure(state=tk.DISABLED)
            self.delete_btn.configure(state=tk.DISABLED)
            
    def start_conversion(self):
        """Start the conversion in a separate thread."""
        if not self.selected_input_file:
            messagebox.showwarning("No File", "Please select an input file first.")
            return
            
        self.processing = True
        self.log_text.delete(1.0, tk.END)
        self.set_status("Processing...", COLORS['warning'])
        self.set_progress(0, "Initializing...")
        
        # Get options
        voxel = self.param_entries["Voxel Size (m):"].get().strip()
        voxel_size = float(voxel) if voxel else None
        knn = int(self.param_entries["KNN Normals:"].get())
        refine = int(self.param_entries["Refine Iterations:"].get())
        remove_outliers = self.outliers_var.get()
        binary = self.format_toggle.get_selected() == "Binary"
        
        # Start thread
        thread = threading.Thread(
            target=self.run_conversion,
            args=(self.selected_input_file, voxel_size, knn, refine, remove_outliers, binary)
        )
        thread.start()
        
    def run_conversion(self, input_file, voxel_size, knn, refine, remove_outliers, binary):
        """Run the conversion process."""
        # Capture output with progress callback
        old_stdout = sys.stdout
        sys.stdout = LogCapture(self.log_text, self._log_callback)
        
        stages = [
            ("Loading...", 10),
            ("Preprocessing...", 25),
            ("Computing normals...", 50),
            ("Building surfels...", 75),
            ("Saving...", 90),
            ("Complete!", 100),
        ]
        
        def update_stage(stage_name, progress):
            def callback():
                self.set_progress(progress, stage_name)
            self.root.after(0, callback)
        
        start_time = time.time()
        current_stage_idx = 0
        
        try:
            update_stage("Loading...", 5)
            print(f"Loading: {input_file.name}")
            points, colors = load_xyzrgb_txt(str(input_file))
            print(f"Loaded {points.shape[0]} points")
            
            update_stage("Preprocessing...", 20)
            print("\n[Preprocessing]")
            
            if remove_outliers and points.shape[0] > 20:
                points, mask = remove_outliers(points, k=20, std_ratio=2.0)
                colors = colors[mask]
                print(f"After outlier removal: {points.shape[0]} points")
                
            points, _ = normalize_points(points)
            print("Normalized")
            
            if voxel_size and voxel_size > 0:
                points, _ = voxel_downsample(points, voxel_size)
                colors = colors[:points.shape[0]]
                print(f"After voxel downsampling: {points.shape[0]} points")
                
            update_stage("Computing normals...", 45)
            print("\n[Normals]")
            k_normal = min(knn, max(3, points.shape[0] - 1))
            normals, curvatures = estimate_normals_knn(points, k=k_normal)
            normals = orient_normals_consistently(points, normals)
            print(f"Estimated {normals.shape[0]} normals")
            print(f"Estimated {normals.shape[0]} normals")
            
            if refine > 0:
                normals = refine_normals(points, normals, iterations=refine)
                print(f"Refined normals ({refine} iterations)")
                
            update_stage("Building surfels...", 70)
            print("\n[Surfels]")
            surfels = build_surfels(points, normals, colors)
            print(f"Built {len(surfels['position'])} surfels")
            
            # Generate output filename with format suffix
            format_suffix = "binary" if binary else "ascii"
            output_name = input_file.stem + f"_{format_suffix}.ply"
            output_file = self.output_dir / output_name
            
            update_stage("Saving...", 90)
            print(f"\n[Export]")
            write_ply(str(output_file), surfels, binary=binary)
            
            elapsed = time.time() - start_time
            print(f"\n{'='*50}")
            print(f"Conversion complete in {elapsed:.2f}s")
            print(f"Output: {output_file.name}")
            print(f"Surfels: {len(surfels['position'])}")
            print(f"Format: {'Binary' if binary else 'ASCII'}")
            print(f"{'='*50}")
            
            update_stage("Complete!", 100)
            self.set_status("Done", COLORS['success'])
            
            # Refresh output list
            self.root.after(0, self.refresh_output_list)
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            self.set_status("Error", COLORS['error'])
            self.set_progress(0, "Error occurred")
            
        finally:
            sys.stdout = old_stdout
            self.processing = False
            self.root.after(0, lambda: self.process_status.configure(
                text=f"Output: {output_file.name}" if 'output_file' in dir() else "Done",
                fg=COLORS['success']
            ))
            
    def _log_callback(self, message):
        """Handle log message for stage detection."""
        if "Loading" in message:
            self.set_progress(10, "Loading...")
        elif "Preprocessing" in message:
            self.set_progress(25, "Preprocessing...")
        elif "Normals" in message:
            self.set_progress(50, "Computing normals...")
        elif "Surfels" in message:
            self.set_progress(75, "Building surfels...")
        elif "Export" in message or "Saving" in message:
            self.set_progress(90, "Saving...")
        elif "complete" in message.lower():
            self.set_progress(100, "Complete!")
            
    def view_selected(self):
        """View the selected PLY file."""
        if not self.selected_output_file:
            return
        
        import subprocess
        viewer_script = str(Path(__file__).parent / "tools" / "viewer.py")
        subprocess.Popen([sys.executable, viewer_script, str(self.selected_output_file)])
        
    def delete_selected(self):
        """Delete the selected output file."""
        if not self.selected_output_file:
            return
            
        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Delete {self.selected_output_file.name}?"
        )
        if confirm:
            self.selected_output_file.unlink()
            self.refresh_output_list()
            self.selected_output_file = None
            self.view_btn.configure(state=tk.DISABLED)
            self.delete_btn.configure(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = ConverterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
