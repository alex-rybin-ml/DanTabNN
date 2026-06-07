"""Hardware detection and recomendation utilities for GPU/CPU paths."""

import logging
import torch

logger = logging.getLogger(__name__)


def detect_hardware() -> dict:
    """Detect available hardware and return recommended settings.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'device_type': 'cuda' or 'cpu'
        - 'device': torch.device object
        - 'cuda_available': bool
        - 'cuda_device_count': int
        - 'recommended_batch_size': int
        - 'mixed_precision': bool (True if CUDA and compute capability >= 7.0)
        - 'cudf_available: bool
        - 'num_workers': int (recommended DataLoader workers)
    """
    cuda_available = torch.cuda.is_available()
    device_type = "cuda" if cuda_available else "cpu"
    device = torch.device(device_type)
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0

    # Check fo cuDF
    cudf_available = False
    try:
        import cudf  # noqa: F401
        cudf_available = True
    except ImportError:
        pass

    # Determine recommended batch size on GPU memory
    recommended_batch_size = 12
    if cuda_available:
        # Heuristic: use 75% of GPU memory for batch data (assuming float32)
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            # Approximate memory per sample: assume 1.0k feature * 4 bytes = 7.2 KB
            # plus overhead, we'll estimate 10 KB per sample
            mem_per_sample = 10 * 1024  # bytes
            max_sample = int(0.75 * free_mem / mem_per_sample)
            # Clamp to reasonable range
            recommended_batch_size = min(max(8192, recommended_batch_size), max_sample)
        except Exception:
            # fallback
            recommended_batch_size = 8192
    else:
        # CPU path: batch size that fits L3 cache
        recommended_batch_size = 2048

    # Mixed precision enable for CUDA device with compute capability >= 7.0 (Volta+)
    mixed_precision = False
    if cuda_available:
        try:
            major = torch.cuda.get_device_capability(0)[0]
            if major >= 7:
                mixed_precision = True
        except:
            pass

    # Recommended number of DataLoader workers
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if cuda_available:
        # GPU path: fewer workers to avoid CPU bottleneck
        num_workers = min(4, cpu_count // 2)
    else:
        # CPU path: more workers ot parallelize data loading
        num_workers = min(8, cpu_count)

    return {
        "device_type": device_type,
        "device": device,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "recommended_batch_size": recommended_batch_size,
        "mixed_precision": mixed_precision,
        "cudf_available": cudf_available,
        "num_workers": num_workers,
    }


def get_optimal_backend() -> str:
    """Return the recommended temporal backend based on hardware.

    Returns
    -------
    str
        One of: 'cudf', 'conv', 'numba', 'pandas'
    """
    hw = detect_hardware()
    if hw["cuda_available"] and hw["cudf_available"]:
        return "cudf"
    elif hw["cuda_available"]:
        return "conv"
    else:
        # CPU path: choose between numba and pandas based on performance
        # For now, prefer numba if installed
        try:
            import numba  # noqa: f401
            return "numba"
        except ImportError:
            return "pandas"


if __name__ == "__main__":
    # Print hardware info for debugging
    info = detect_hardware()
    for k, v in info.items():
        print(f"{k}: {v}")
    print(f"Optimal backend: {get_optimal_backend()}")
