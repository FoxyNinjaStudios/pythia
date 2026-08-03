"""
Device utility for SAM-3D CPU/MPS compatibility.
"""
import torch


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS available but spconv only supports CPU, so use CPU
        return torch.device("cpu")
    return torch.device("cpu")


# Default device for this installation
DEVICE = get_device()
