#!/usr/bin/env python3
"""
LiDAR to 2DGS Converter GUI (Survey Grade)

A graphical interface for converting LiDAR point clouds to 2DGS surfels.
Similar to pc_to_2dgs GUI but with survey-grade options.

Usage:
    python gui2.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import PIL for image handling
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Note: PIL not installed. Logo will not be displayed.")

# Try to import dependencies
try:
    import numpy as np
except ImportError:
    print("Error: numpy not installed. Please run: pip install numpy")

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("Error: scipy not installed. Please run: pip install scipy")

# Color scheme (same as pc_to_2dgs)
COLORS = {
    'bg': '#0F172A',
    'card_bg': '#111827',
    'border': '#1F2937',
    'accent': '#3B82F6',
    'accent_hover': '#2563EB',
    'success': '#22C55E',
    'warning': '#F59E0B',
    'error': '#EF4444',
    'text': '#F9FAFB',
    'text_secondary': '#9CA3AF',
    'input_bg': '#1F2937',
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


class ConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LiDAR to 2DGS Converter (Survey Grade)")
        self.root.geometry("1100x850")
        self.root.minsize(1000, 750)
        self.root.configure(bg=COLORS['bg'])
        
        # Project paths
        self.project_dir = Path(__file__).parent
        self.input_dir = self.project_dir / "data" / "input"
        self.output_dir = self.project_dir / "data" / "output"
        
        # State
        self.selected_input_file = None
        self.selected_output_file = None
        self.processing = False
        
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
        
        style.configure('Card.TFrame', background=COLORS['card_bg'])
        style.configure('TFrame', background=COLORS['bg'])
        style.configure('TLabel', background=COLORS['card_bg'], foreground=COLORS['text'])
        style.configure('TLabelframe', background=COLORS['card_bg'], 
                       foreground=COLORS['text'])
        style.configure('TLabelframe.Label', background=COLORS['card_bg'], 
                       foreground=COLORS['text'], font=('Segoe UI', 10, 'bold'))
        
        style.configure('TEntry', fieldbackground=COLORS['input_bg'], 
                       foreground=COLORS['text'], borderwidth=0)
        style.map('TEntry', fieldbackground=[('focus', COLORS['accent'])])
        
        style.configure('Horizontal.TProgressbar', 
                       troughcolor=COLORS['border'],
                       background=COLORS['accent'],
                       thickness=6)
        
        style.configure('Vertical.TScrollbar', 
                       troughcolor=COLORS['bg'],
                       background=COLORS['border'],
                       arrowcolor=COLORS['text'])
        
    def setup_ui(self):
        """Create the GUI layout."""
        # ==== HEADER ====
        header = tk.Frame(self.root, bg=COLORS['card_bg'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # Logo
        logo_path = self.project_dir / "HDB4.png"
        if HAS_PIL and logo_path.exists():
            try:
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((80, 80), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                
                logo_label = tk.Label(header, image=self.logo_photo, bg=COLORS['card_bg'])
                logo_label.image = self.logo_photo
                logo_label.pack(side=tk.LEFT, padx=16, pady=10)
            except Exception as e:
                print(f"Warning: Could not load logo: {e}")
        
        # Title
        title_label = tk.Label(header, text="LiDAR to 2DGS (Survey Grade)", 
                              bg=COLORS['card_bg'], fg=COLORS['text'],
                              font=('Segoe UI', 16, 'bold'))
        title_label.pack(side=tk.LEFT, pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(header, text="Point cloud to Gaussian surfels", 
                                 bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                                 font=('Segoe UI', 10))
        subtitle_label.pack(side=tk.LEFT, pady=(28, 10))
        
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
        
        # Three-column layout
        left_col = tk.Frame(main_frame, bg=COLORS['bg'], width=300)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_col.pack_propagate(False)
        
        mid_col = tk.Frame(main_frame, bg=COLORS['bg'], width=350)
        mid_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 12))
        mid_col.pack_propagate(False)
        
        right_col = tk.Frame(main_frame, bg=COLORS['bg'], width=280)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_col.pack_propagate(False)
        
        # ==== LEFT COLUMN: Input Files ====
        self.create_card(left_col, "Input Files")
        input_frame = self.current_card
        
        list_container = tk.Frame(input_frame, bg=COLORS['card_bg'], height=120)
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
        
        self.create_header_button(btn_row, "↻ Refresh", self.refresh_input_list).pack(side=tk.LEFT, padx=8)
        self.create_header_button(btn_row, "+ Add File...", self.add_input_file).pack(side=tk.RIGHT, padx=8)
        
        # ==== MIDDLE COLUMN: Processing Options ====
        self.create_card(mid_col, "Processing Options")
        options_frame = self.current_card
        
        # Preprocessing section
        self.create_section_header(options_frame, "Preprocessing")
        
        # Parameters
        params = [
            ("Voxel Size (m):", "", "Leave empty to disable voxel downsampling"),
            ("Outlier Threshold:", "2.0", "Z-score threshold for outlier removal (0 to disable)"),
            ("Outlier K:", "20", "Neighbors for outlier detection"),
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
                           width=12)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT)
            self.param_entries[label_text] = entry
        
        # Downsampling checkbox
        self.downsample_var = tk.BooleanVar(value=False)
        downsample_cb = tk.Checkbutton(options_frame, text="Enable Auto Downsampling (100:1)",
                                      variable=self.downsample_var,
                                      bg=COLORS['card_bg'], fg=COLORS['text'],
                                      selectcolor=COLORS['card_bg'],
                                      activebackground=COLORS['card_bg'],
                                      font=('Segoe UI', 9), cursor='hand2')
        downsample_cb.pack(anchor=tk.W, padx=8, pady=4)
        
        # Normal Estimation section
        self.create_section_header(options_frame, "Normal Estimation")
        
        params2 = [
            ("K Neighbors:", "20", "Number of neighbors for normal estimation"),
            ("Sigma Tangent (m):", "0.05", "Spread along tangent plane"),
            ("Sigma Normal (m):", "0.002", "Thickness along normal (0.01-0.02 for denser output)"),
            ("Opacity:", "0.8", "Default opacity (0.1-1.0)"),
        ]
        
        for label_text, default, tooltip in params2:
            row = tk.Frame(options_frame, bg=COLORS['card_bg'])
            row.pack(fill=tk.X, padx=12, pady=2)
            
            lbl = tk.Label(row, text=label_text, 
                          bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                          font=('Segoe UI', 9), width=18, anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            
            entry = tk.Entry(row, bg=COLORS['input_bg'], fg=COLORS['text'],
                           font=('Consolas', 10), bd=0, highlightthickness=1,
                           highlightbackground=COLORS['border'],
                           width=12)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT)
            self.param_entries[label_text] = entry
        
        # Quality options
        self.create_section_header(options_frame, "Quality Options")
        
        self.uncertainty_var = tk.BooleanVar()
        uncertainty_cb = tk.Checkbutton(options_frame, text="Enable Uncertainty Quantification",
                                       variable=self.uncertainty_var,
                                       bg=COLORS['card_bg'], fg=COLORS['text'],
                                       selectcolor=COLORS['card_bg'],
                                       activebackground=COLORS['card_bg'],
                                       font=('Segoe UI', 9), cursor='hand2')
        uncertainty_cb.pack(anchor=tk.W, padx=8, pady=2)
        
        self.validate_var = tk.BooleanVar()
        validate_cb = tk.Checkbutton(options_frame, text="Validate Normals Against Points",
                                     variable=self.validate_var,
                                     bg=COLORS['card_bg'], fg=COLORS['text'],
                                     selectcolor=COLORS['card_bg'],
                                     activebackground=COLORS['card_bg'],
                                     font=('Segoe UI', 9), cursor='hand2')
        validate_cb.pack(anchor=tk.W, padx=8, pady=2)
        
        # Device selection
        self.create_section_header(options_frame, "Device")
        
        device_frame = tk.Frame(options_frame, bg=COLORS['card_bg'])
        device_frame.pack(fill=tk.X, padx=8, pady=4)
        
        self.device_var = tk.StringVar(value="auto")
        
        rb1 = tk.Radiobutton(device_frame, text="Auto (GPU if available)", 
                             variable=self.device_var, value="auto",
                             bg=COLORS['card_bg'], fg=COLORS['text'],
                             selectcolor=COLORS['card_bg'],
                             font=('Segoe UI', 9), cursor='hand2')
        rb1.pack(anchor=tk.W, padx=4)
        
        rb2 = tk.Radiobutton(device_frame, text="CPU Only", 
                             variable=self.device_var, value="cpu",
                             bg=COLORS['card_bg'], fg=COLORS['text'],
                             selectcolor=COLORS['card_bg'],
                             font=('Segoe UI', 9), cursor='hand2')
        rb2.pack(anchor=tk.W, padx=4)
        
        # ==== RIGHT COLUMN: Run & Viewer ====
        
        # Run Card
        self.create_card(right_col, "Run")
        run_frame = self.current_card
        
        # Progress
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
        
        # Output Card
        self.create_card(right_col, "Output Files")
        output_frame = self.current_card
        
        list_container2 = tk.Frame(output_frame, bg=COLORS['card_bg'], height=100)
        list_container2.pack(fill=tk.X, pady=(8, 8))
        list_container2.pack_propagate(False)
        
        self.output_listbox = tk.Listbox(list_container2, selectmode=tk.SINGLE,
                                        font=('Consolas', 10),
                                        bg=COLORS['input_bg'], fg=COLORS['text'],
                                        bd=0, highlightthickness=1,
                                        highlightbackground=COLORS['border'],
                                        selectbackground=COLORS['accent'],
                                        selectforeground='#FFFFFF')
        scrollbar2 = ttk.Scrollbar(list_container2, orient=tk.VERTICAL,
                                   command=self.output_listbox.yview)
        self.output_listbox.config(yscrollcommand=scrollbar2.set)
        
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self.output_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.output_listbox.bind('<<ListboxSelect>>', self.on_output_selected)
        
        # Buttons
        btn_row2 = tk.Frame(output_frame, bg=COLORS['card_bg'])
        btn_row2.pack(fill=tk.X, pady=(0, 8))
        
        self.create_header_button(btn_row2, "Refresh", self.refresh_output_list).pack(side=tk.LEFT, padx=8)
        
        self.view_btn = ttk.Button(btn_row2, text="View", 
                                   command=self.view_selected,
                                   state=tk.DISABLED)
        self.view_btn.pack(side=tk.RIGHT, padx=8)
        
        self.delete_btn = ttk.Button(btn_row2, text="Delete",
                                    command=self.delete_selected,
                                    state=tk.DISABLED)
        self.delete_btn.pack(side=tk.RIGHT, padx=4)
        
        # Logs Card
        self.create_card(right_col, "Logs")
        log_frame = self.current_card
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8,
                                                 font=('Consolas', 9),
                                                 bg=COLORS['input_bg'], 
                                                 fg=COLORS['text'],
                                                 bd=0, highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
    def create_card(self, parent, title):
        """Create a card container."""
        card = tk.Frame(parent, bg=COLORS['card_bg'], bd=1, relief=tk.SOLID,
                       highlightbackground=COLORS['border'], highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 12))
        self.current_card = card
        
        title_label = tk.Label(card, text=title, bg=COLORS['card_bg'], 
                              fg=COLORS['text'], font=('Segoe UI', 10, 'bold'),
                              padx=12, pady=8)
        title_label.pack(anchor=tk.W)
        
        return card
    
    def create_section_header(self, parent, text):
        """Create a section header."""
        header = tk.Label(parent, text=text, bg=COLORS['card_bg'], 
                         fg=COLORS['accent'], font=('Segoe UI', 9, 'bold'))
        header.pack(anchor=tk.W, padx=12, pady=(8, 4))
        
    def create_header_button(self, parent, text, command):
        """Create a header-style button."""
        btn = tk.Label(parent, text=text, bg=COLORS['card_bg'],
                      fg=COLORS['accent'], font=('Segoe UI', 9),
                      cursor='hand2', padx=8, pady=4)
        btn.bind("<Button-1>", lambda e: command())
        btn.bind("<Enter>", lambda e: btn.configure(fg=COLORS['accent_hover']))
        btn.bind("<Leave>", lambda e: btn.configure(fg=COLORS['accent']))
        return btn
    
    def set_status(self, status, color=None):
        """Update status indicator and label."""
        self.status_label.configure(text=status)
        
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
        """Format file size in human-readable format."""
        if size_bytes >= 1_000_000_000:
            return f"{size_bytes / 1_000_000_000:.2f} GB"
        elif size_bytes >= 1_000_000:
            return f"{size_bytes / 1_000_000:.2f} MB"
        elif size_bytes >= 1_000:
            return f"{size_bytes / 1_000:.2f} KB"
        else:
            return f"{size_bytes} B"
    
    def refresh_input_list(self):
        """Refresh the list of input files."""
        self.input_listbox.delete(0, tk.END)
        
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # Support TXT, LAS, LAZ files
        files = []
        files.extend(self.input_dir.glob("*.txt"))
        files.extend(self.input_dir.glob("*.las"))
        files.extend(self.input_dir.glob("*.laz"))
        files = sorted(set(files))
        
        for f in files:
            size_str = self.format_file_size(f.stat().st_size)
            ext = f.suffix.upper()
            self.input_listbox.insert(tk.END, f"{f.name} ({size_str}) [{ext}]")
        
        if not files:
            self.input_listbox.insert(tk.END, "(No supported files found)")
    
    def refresh_output_list(self):
        """Refresh the list of output files."""
        self.output_listbox.delete(0, tk.END)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for both .ply files and .2dgs_octree directories
        # The viewer supports both formats
        ply_files = sorted(self.output_dir.glob("*.ply"))
        octree_dirs = sorted(self.output_dir.glob("*.2dgs_octree"))
        
        # Calculate total size for .2dgs_octree directories
        def get_dir_size(path):
            total = 0
            for f in path.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
            return total
        
        # Show PLY files
        for f in ply_files:
            size_str = self.format_file_size(f.stat().st_size)
            self.output_listbox.insert(tk.END, f"{f.name} ({size_str})")
        
        # Show .2dgs_octree directories (streaming format)
        for d in octree_dirs:
            size_str = self.format_file_size(get_dir_size(d))
            self.output_listbox.insert(tk.END, f"{d.name}/ ({size_str})")
        
        if not ply_files and not octree_dirs:
            self.output_listbox.insert(tk.END, "(No output files found)")
    
    def add_input_file(self):
        """Open file dialog to add a new input file."""
        filename = filedialog.askopenfilename(
            title="Select Point Cloud",
            filetypes=[
                ("Point clouds", "*.txt *.las *.laz"),
                ("Text files", "*.txt"),
                ("LAS files", "*.las"),
                ("All files", "*.*")
            ]
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
            # Extract filename (remove size and format info)
            # Format: "filename.ext (size) [FORMAT]"
            filename = text.split(" (")[0]
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
        try:
            voxel_size_str = self.param_entries["Voxel Size (m):"].get().strip()
            voxel_size = float(voxel_size_str) if voxel_size_str else None
            
            outlier_threshold = float(self.param_entries["Outlier Threshold:"].get())
            outlier_k = int(self.param_entries["Outlier K:"].get())
            k_neighbors = int(self.param_entries["K Neighbors:"].get())
            sigma_tangent = float(self.param_entries["Sigma Tangent (m):"].get())
            sigma_normal = float(self.param_entries["Sigma Normal (m):"].get())
            opacity = float(self.param_entries["Opacity:"].get())
            
            enable_downsample = self.downsample_var.get()
            enable_uncertainty = self.uncertainty_var.get()
            enable_validate = self.validate_var.get()
            device = self.device_var.get()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid parameter value: {e}")
            self.processing = False
            return
        
        # Start thread
        thread = threading.Thread(
            target=self.run_conversion,
            args=(
                self.selected_input_file, voxel_size, outlier_threshold, outlier_k,
                k_neighbors, sigma_tangent, sigma_normal, opacity,
                enable_downsample, enable_uncertainty, enable_validate, device
            )
        )
        thread.start()
    
    def run_conversion(self, input_file, voxel_size, outlier_threshold, outlier_k,
                       k_neighbors, sigma_tangent, sigma_normal, opacity,
                       enable_downsample, enable_uncertainty, enable_validate, device):
        """Run the conversion process."""
        old_stdout = sys.stdout
        sys.stdout = LogCapture(self.log_text, self._log_callback)
        
        import numpy as np
        from scipy.spatial import cKDTree
        
        try:
            print(f"{'='*60}")
            print(f"  LiDAR to 2DGS Conversion")
            print(f"{'='*60}")
            print(f"Input: {input_file.name}")
            
            # Determine device
            if device == 'auto':
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_name = 'CUDA'
                    else:
                        device_name = 'CPU'
                except ImportError:
                    device_name = 'CPU'
            else:
                device_name = 'CPU'
            
            print(f"Device: {device_name}")
            
            # STEP 1: Load Data
            print(f"\n[STEP 1] Loading Data")
            ext = input_file.suffix.lower()
            
            if ext == '.las':
                from src.las_io import load_las
                data = load_las(str(input_file))
                points = data["position"]
                colors = data.get("color")
                if colors is not None and colors.dtype == np.uint8:
                    pass  # Keep as uint8
            elif ext == '.laz':
                from src.las_io import load_las
                data = load_las(str(input_file))
                points = data["position"]
                colors = data.get("color")
            else:
                from src.txt_io import load_xyzrgb_txt
                data = load_xyzrgb_txt(str(input_file))
                points = data["position"]
                colors = data.get("color")
            
            original_count = points.shape[0]
            print(f"Points loaded: {original_count:,}")
            
            # STEP 2: Preprocessing
            print(f"\n[STEP 2] Preprocessing")
            
            # Outlier removal
            if outlier_threshold > 0:
                from src.preprocess import remove_outliers_statistical
                result = remove_outliers_statistical(
                    points, k=outlier_k, std_multiplier=outlier_threshold
                )
                points = result["position"]
                if colors is not None:
                    colors = colors[result["mask"]]
                print(f"Outliers removed: {result['removed_count']:,}")
            
            # Downsampling
            if enable_downsample:
                from src.preprocess import voxel_downsample, calculate_voxel_size_for_ratio
                auto_voxel_size = calculate_voxel_size_for_ratio(points, target_ratio=100.0)
                voxel_result = voxel_downsample(points, voxel_size=auto_voxel_size, colors=colors)
                points = voxel_result["position"]
                colors = voxel_result.get("color")
                reduction = 100 * (1 - len(points) / original_count)
                print(f"Auto downsampling: {reduction:.1f}% reduction")
                print(f"Voxel size: {auto_voxel_size:.4f}m")
            elif voxel_size and voxel_size > 0:
                from src.preprocess import voxel_downsample
                voxel_result = voxel_downsample(points, voxel_size=voxel_size, colors=colors)
                points = voxel_result["position"]
                colors = voxel_result.get("color")
                print(f"Voxel downsampling enabled: {voxel_size}m")
            
            final_count = points.shape[0]
            print(f"Final points: {final_count:,}")
            
            # STEP 3: Normal Estimation
            print(f"\n[STEP 3] Normal Estimation")
            up_vector = (0.0, 0.0, 1.0)
            
            if enable_uncertainty:
                from src.normals import estimate_normals_with_uncertainty
                normal_data = estimate_normals_with_uncertainty(
                    points, k_neighbors=k_neighbors, up_vector=up_vector, device=device
                )
                normals = normal_data["normals"]
                uncertainty = normal_data["uncertainty"]
                print(f"Mean uncertainty: {uncertainty.mean():.4f}")
            else:
                from src.normals import estimate_normals_knn
                normals = estimate_normals_knn(
                    points, k_neighbors=k_neighbors, up_vector=up_vector, device=device
                )
            
            print(f"Normals computed: {normals.shape[0]:,}")
            
            # Validation
            if enable_validate:
                from src.normals import validate_normals_against_points
                validation = validate_normals_against_points(normals, points)
                print(f"Mean consistency: {validation['consistency'].mean():.4f}")
            
            # STEP 4: Surfel Construction
            print(f"\n[STEP 4] Surfel Construction")
            from src.surfels import build_surfels
            
            surfels = build_surfels(
                points, normals, colors=colors,
                sigma_tangent=sigma_tangent,
                sigma_normal=sigma_normal,
                opacity=opacity
            )
            print(f"Surfels created: {surfels['position'].shape[0]:,}")
            
            # STEP 5: Export
            print(f"\n[STEP 5] Export")
            output_file = self.output_dir / f"{input_file.stem}_2dgs.ply"
            
            from src.export_ply import write_ply
            write_ply(str(output_file), surfels, binary=True)
            
            print(f"Output: {output_file.name}")
            print(f"\n{'='*60}")
            print(f"  Complete!")
            print(f"{'='*60}")
            
            self.set_progress(100, "Complete!")
            self.set_status("Done", COLORS['success'])
            
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
    
    def _log_callback(self, message):
        """Handle log message for stage detection."""
        if "Loading" in message:
            self.set_progress(10, "Loading...")
        elif "Preprocessing" in message:
            self.set_progress(25, "Preprocessing...")
        elif "Normal" in message:
            self.set_progress(50, "Computing normals...")
        elif "Surfel" in message:
            self.set_progress(75, "Building surfels...")
        elif "Export" in message or "Saving" in message:
            self.set_progress(90, "Saving...")
        elif "Complete" in message:
            self.set_progress(100, "Complete!")
    
    def view_selected(self):
        """View the selected PLY file or octree directory using the OpenGL streaming viewer."""
        if not self.selected_output_file:
            return
        
        import subprocess
        import sys
        from pathlib import Path
        
        # Check if it's a .2dgs_octree directory or a .ply file
        is_octree = self.selected_output_file.suffix == '.2dgs_octree'
        is_ply = self.selected_output_file.suffix == '.ply'
        
        # If it's a directory, check if it's an octree directory
        if self.selected_output_file.is_dir():
            is_octree = self.selected_output_file.name.endswith('.2dgs_octree')
        
        # Use simple_viewer.py for regular PLY files (smaller files)
        # Use streaming_viewer_main.py for .2dgs_octree directories (large files)
        if is_octree:
            # Use streaming viewer for octree directories (handles large files)
            viewer_script = self.project_dir / "tools" / "streaming_viewer_main.py"
            viewer_type = "streaming"
        elif is_ply:
            # Check file size to decide which viewer to use
            file_size = self.selected_output_file.stat().st_size
            if file_size > 500 * 1024 * 1024:  # > 500MB
                # For large PLY files, try to find/create octree first
                messagebox.showinfo("Large File", 
                    f"This PLY file is {file_size / (1024*1024):.1f}MB. "
                    "For best performance, convert to .2dgs_octree format first.\n"
                    "Attempting to view with streaming viewer...")
                viewer_script = self.project_dir / "tools" / "streaming_viewer_main.py"
                viewer_type = "streaming"
            else:
                # Use simple viewer for smaller files
                viewer_script = self.project_dir / "simple_viewer.py"
                viewer_type = "simple"
        else:
            # Try streaming viewer as fallback
            viewer_script = self.project_dir / "tools" / "streaming_viewer_main.py"
            viewer_type = "streaming"
        
        if viewer_script.exists():
            print(f"Launching {viewer_type} viewer for: {self.selected_output_file.name}")
            
            # Run viewer as subprocess
            try:
                subprocess.Popen(
                    [sys.executable, str(viewer_script), str(self.selected_output_file)],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            except Exception as e:
                # Fallback without console creation
                try:
                    subprocess.Popen(
                        [sys.executable, str(viewer_script), str(self.selected_output_file)]
                    )
                except Exception as e2:
                    messagebox.showerror("Viewer Error", 
                        f"Failed to launch viewer: {e2}")
        else:
            messagebox.showerror("Viewer Error", 
                f"Viewer not found at: {viewer_script}")
    
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
