"""
Server-side AI mesh cleanup model management and inference.
Handles loading, unloading, and running PCN denoising + Shape VAE/SnowflakeNet completion on Metal/GPU.
Supports downloading pretrained weights from Hugging Face and Google Drive.
"""

import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Model registry with download information
AI_CLEANUP_MODELS = {
    "pcn-denoise": {
        "name": "Point Completion Network (Denoising)",
        "description": "Denoise point clouds using Point Completion Network (PCN). Removes noise artifacts from reconstructed geometry.",
        "size_mb": 85,
        "source": "huggingface",
        "url": "https://huggingface.co/FoxyNinjaStudios/pythia-ai-cleanup/resolve/main/pcn_model.pt",
        "license": "MIT (wentaoyuan/pcn PyTorch port)",
    },
    "snowflakenet": {
        "name": "SnowflakeNet (Snowflake Point Deconvolution)",
        "description": "Advanced point cloud completion using Snowflake Point Deconvolution with Skip-Transformer. Fills holes and refines geometry with learned patterns (ICCV 2021, TPAMI 2023).",
        "size_mb": 350,
        "source": "gdrive",
        "gdrive_folder_id": "1mdA-6ZwzXAbaWJ6fmfL9-gl3aGTGTWyR",
        "gdrive_model_file": "spd_scannet_mix.ckpt",
        "license": "MIT (AllenXiangX/SnowflakeNet)",
        "paper": "https://arxiv.org/abs/2202.09367",
    },
    "shape-vae": {
        "name": "3D Shape VAE",
        "description": "Complete and refine 3D shapes using a Variational Autoencoder trained on ShapeNet. Fills holes and improves geometry coherence.",
        "size_mb": 120,
        "source": "huggingface",
        "url": "https://huggingface.co/FoxyNinjaStudios/pythia-ai-cleanup/resolve/main/shape_vae_model.pt",
        "license": "MIT (autonomousvision/occupancy-networks)",
    },
}

# Global state
_ai_models_loaded = {}
_ai_models_lock = threading.Lock()


def _get_weight_status(model_id: str) -> str:
    """Check if model weights are downloaded."""
    weight_path = Path("checkpoints/ai_cleanup") / f"{model_id.split('-')[0]}_model.pt"
    if weight_path.exists():
        return "downloaded"
    return "missing"


def get_ai_model_status() -> Dict:
    """Get status of all AI cleanup models (loaded/downloaded/missing)."""
    with _ai_models_lock:
        status = {}
        for model_id, meta in AI_CLEANUP_MODELS.items():
            is_loaded = model_id in _ai_models_loaded and _ai_models_loaded[model_id] is not None
            weight_status = _get_weight_status(model_id)
            
            if is_loaded:
                dl_status = "loaded"
            elif weight_status == "downloaded":
                dl_status = "downloaded"
            else:
                dl_status = "missing"
            
            status[model_id] = {
                "name": meta["name"],
                "description": meta["description"],
                "size_mb": meta["size_mb"],
                "status": dl_status,
                "loaded": is_loaded,
                "source": meta["source"],
                "license": meta.get("license", "Unknown"),
            }
        return status


def download_ai_model(model_id: str) -> bool:
    """Download an AI model's pretrained weights. Returns True if successful."""
    if model_id not in AI_CLEANUP_MODELS:
        logger.error(f"Unknown AI model: {model_id}")
        return False
    
    try:
        import mesh_cleanup_ai
        
        meta = AI_CLEANUP_MODELS[model_id]
        
        if meta["source"] == "gdrive":
            # Download from Google Drive using gdown
            mesh_model_name = "snowflakenet"
            logger.info(f"Downloading {model_id} weights from Google Drive…")
            weight_path = mesh_cleanup_ai._download_gdrive_weight(
                mesh_model_name,
                meta["gdrive_folder_id"],
                meta["gdrive_model_file"]
            )
        else:
            # Download from Hugging Face
            mesh_model_name = "pcn" if model_id == "pcn-denoise" else "shape_vae"
            url = meta["url"]
            logger.info(f"Downloading {model_id} weights from {meta['source']}…")
            weight_path = mesh_cleanup_ai._download_weight(mesh_model_name, url)
        
        return weight_path is not None and weight_path.exists()
    
    except Exception as e:
        logger.error(f"Failed to download {model_id}: {e}")
        return False


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
            
            elif model_id == "snowflakenet":
                logger.info("Loading SnowflakeNet completion model…")
                model = mesh_cleanup_ai.load_shape_vae_model()
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
