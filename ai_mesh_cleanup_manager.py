"""
Server-side AI mesh cleanup model management and inference.
Handles loading, unloading, and running PCN denoising + Shape VAE completion on Metal/GPU.
"""

import os
import threading
from typing import Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Model registry
AI_CLEANUP_MODELS = {
    "pcn-denoise": {
        "name": "Point Completion Network (Denoising)",
        "description": "Denoise point clouds using PCN",
        "size_mb": 50,
        "source": "builtin",
    },
    "shape-vae": {
        "name": "3D Shape VAE",
        "description": "Complete/refine 3D shapes using variational autoencoder",
        "size_mb": 80,
        "source": "builtin",
    },
}

# Global state
_ai_models_loaded = {}
_ai_models_lock = threading.Lock()


def get_ai_model_status() -> Dict:
    """Get status of all AI cleanup models."""
    with _ai_models_lock:
        status = {}
        for model_id, meta in AI_CLEANUP_MODELS.items():
            is_loaded = model_id in _ai_models_loaded and _ai_models_loaded[model_id] is not None
            status[model_id] = {
                "name": meta["name"],
                "description": meta["description"],
                "size_mb": meta["size_mb"],
                "loaded": is_loaded,
                "source": meta["source"],
            }
        return status


def load_ai_model(model_id: str) -> bool:
    """Load an AI cleanup model into memory. Returns True if successful."""
    if model_id not in AI_CLEANUP_MODELS:
        logger.error(f"Unknown AI model: {model_id}")
        return False
    
    with _ai_models_lock:
        if model_id in _ai_models_loaded and _ai_models_loaded[model_id] is not None:
            logger.info(f"AI model {model_id} already loaded")
            return True
        
        try:
            import mesh_cleanup_ai
            
            if model_id == "pcn-denoise":
                logger.info("Loading PCN denoising model…")
                model = mesh_cleanup_ai.load_pcn_model()
                _ai_models_loaded[model_id] = model
                return model is not None
            
            elif model_id == "shape-vae":
                logger.info("Loading Shape VAE model…")
                model = mesh_cleanup_ai.load_shape_vae_model()
                _ai_models_loaded[model_id] = model
                return model is not None
        
        except Exception as e:
            logger.error(f"Failed to load AI model {model_id}: {e}")
            _ai_models_loaded[model_id] = None
            return False


def unload_ai_model(model_id: str) -> bool:
    """Unload an AI model to free GPU/Metal memory."""
    with _ai_models_lock:
        if model_id in _ai_models_loaded:
            del _ai_models_loaded[model_id]
            try:
                import mesh_cleanup_ai
                mesh_cleanup_ai.unload_models()
                logger.info(f"AI model {model_id} unloaded")
                return True
            except Exception as e:
                logger.warning(f"Error unloading AI model {model_id}: {e}")
                return False
    return False


def unload_all_ai_models():
    """Unload all AI models."""
    with _ai_models_lock:
        try:
            import mesh_cleanup_ai
            mesh_cleanup_ai.unload_models()
            _ai_models_loaded.clear()
            logger.info("All AI models unloaded")
        except Exception as e:
            logger.warning(f"Error unloading AI models: {e}")


def apply_ai_cleanup(
    vertices: np.ndarray,
    faces: np.ndarray,
    enable_denoising: bool = False,
    enable_completion: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply AI-based mesh cleanup.
    
    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) mesh faces
        enable_denoising: Apply PCN point cloud denoising
        enable_completion: Apply VAE shape completion
        verbose: Print progress
    
    Returns:
        (vertices, faces) cleaned/completed mesh
    """
    if not enable_denoising and not enable_completion:
        return vertices, faces
    
    try:
        import mesh_cleanup_ai
        
        # Load required models
        if enable_denoising and not load_ai_model("pcn-denoise"):
            logger.warning("PCN model failed to load, skipping denoising")
            enable_denoising = False
        
        if enable_completion and not load_ai_model("shape-vae"):
            logger.warning("Shape VAE model failed to load, skipping completion")
            enable_completion = False
        
        # Apply cleanup
        return mesh_cleanup_ai.denoise_and_complete_mesh(
            vertices, faces,
            enable_denoising=enable_denoising,
            enable_completion=enable_completion,
            verbose=verbose,
        )
    
    except Exception as e:
        logger.error(f"AI mesh cleanup failed: {e}")
        return vertices, faces


def cleanup_ai_on_shutdown():
    """Clean up AI models on server shutdown."""
    logger.info("Cleaning up AI mesh cleanup models…")
    unload_all_ai_models()
