# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Memory management utilities for SAM-3D pipeline."""

import gc
import torch
from loguru import logger


def offload_model(model, name: str = "model"):
    """Offload a model from memory by clearing all parameters and deleting."""
    if model is None:
        return
    
    try:
        # Move to CPU first
        model.cpu()
        
        # Clear all parameters to free memory
        for param in model.parameters():
            param.data = torch.empty(0)
            if param.grad is not None:
                param.grad = None
        
        # Clear all buffers
        for buffer_name, buffer in model.named_buffers():
            buffer.data = torch.empty(0)
        
        logger.info(f"Offloaded and cleared {name}")
    except Exception as e:
        logger.warning(f"Failed to offload {name}: {e}")



def delete_model(model, name: str = "model"):
    """Fully delete a model and free its memory."""
    if model is None:
        return
    
    try:
        # Move to CPU first
        model.cpu()
        # Delete the model
        del model
        logger.info(f"Deleted {name}")
    except Exception as e:
        logger.warning(f"Failed to delete {name}: {e}")


def clear_memory():
    """Force garbage collection and clear PyTorch caches."""
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Clear MPS cache if available (macOS)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing, but we can force sync
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    
    # Force another GC pass
    gc.collect()


def get_memory_usage_mb():
    """Get current memory usage in MB (approximate)."""
    import os
    
    # Try to get RSS from /proc on Linux
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # Convert KB to MB
    except:
        pass
    
    # On macOS, use resource module
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 * 1024)  # Convert bytes to MB on macOS
    except:
        pass
    
    return -1


def print_memory_usage(stage: str = ""):
    """Print current memory usage for debugging."""
    mem_mb = get_memory_usage_mb()
    if mem_mb > 0:
        logger.info(f"[Memory] {stage}: {mem_mb:.1f} MB")
    else:
        logger.info(f"[Memory] {stage}: (unable to measure)")
