"""Tests for GPU functionality (skip if CUDA unavailable)."""

import numpy as np
import pytest

# Check for CUDA availability
try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Pytest marker for GPU tests
pytestmark = pytest.mark.skipif(
    not HAS_CUDA or not HAS_FAISS,
    reason="CUDA and FAISS required for GPU tests"
)


class TestGPUNormals:
    """Tests for GPU-accelerated normal estimation."""
    
    @pytest.fixture
    def sample_points(self):
        """Generate sample point cloud."""
        np.random.seed(42)
        return np.random.randn(50000, 3).astype(np.float32) * 10
    
    def test_gpu_faiss_normals(self, sample_points):
        """Test GPU FAISS normal estimation."""
        from src.normals import estimate_normals_knn
        
        normals = estimate_normals_knn(sample_points, k_neighbors=20, device='cuda')
        
        assert normals.shape == sample_points.shape
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
        assert np.isfinite(normals).all()
    
    def test_gpu_faiss_large_dataset(self):
        """Test GPU FAISS on larger dataset."""
        from src.normals import estimate_normals_knn
        
        # Generate larger dataset
        points = np.random.randn(200000, 3).astype(np.float32) * 10
        
        normals = estimate_normals_knn(points, k_neighbors=20, device='cuda')
        
        assert normals.shape == points.shape
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    
    def test_gpu_uncertainty(self, sample_points):
        """Test GPU uncertainty quantification."""
        from src.normals import estimate_normals_with_uncertainty
        
        result = estimate_normals_with_uncertainty(sample_points, device='cuda')
        
        assert "normals" in result
        assert "uncertainty" in result
        assert "planarity" in result
        assert result["normals"].shape == sample_points.shape
        assert np.all(result["uncertainty"] >= 0)
        assert np.all(result["uncertainty"] <= 1)
    
    def test_gpu_memory_cleanup(self, sample_points):
        """Test that GPU memory is properly cleaned up."""
        from src.normals import estimate_normals_knn
        import torch
        
        # Clear cache before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Run estimation
        normals = estimate_normals_knn(sample_points, k_neighbors=20, device='cuda')
        
        # Clear cache after
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            
            # Memory should be cleaned up (allow some tolerance)
            assert final_memory <= initial_memory + 10 * 1024 * 1024  # 10MB tolerance


class TestGPUDeviceSelection:
    """Tests for GPU device selection and fallback."""
    
    def test_cuda_device_requested(self):
        """Test that CUDA device can be requested."""
        from src.normals import estimate_normals_knn
        
        points = np.random.randn(1000, 3).astype(np.float32)
        
        # Should work if CUDA available, otherwise raise
        try:
            normals = estimate_normals_knn(points, device='cuda')
            assert normals.shape == points.shape
        except RuntimeError as e:
            # Expected if CUDA not available
            assert "CUDA" in str(e) or "not available" in str(e)
    
    def test_auto_device_selection(self):
        """Test automatic device selection."""
        from src.normals import estimate_normals_knn, get_device
        
        points = np.random.randn(1000, 3).astype(np.float32)
        device = get_device()
        
        # Should work with auto-detected device
        normals = estimate_normals_knn(points, device=device)
        assert normals.shape == points.shape
