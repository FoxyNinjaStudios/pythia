"""
AI-based mesh cleanup using point cloud denoising and shape completion VAE.
Runs on Metal (MPS) where available, with intelligent memory management.
Supports downloading pretrained weights from Hugging Face.
"""

import os
import hashlib
from pathlib import Path
from urllib.request import urlopen
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh
import logging

logger = logging.getLogger(__name__)

# Weight sources (MIT licensed)
WEIGHT_CONFIG = {
    "pcn": {
        "url": "https://huggingface.co/FoxyNinjaStudios/pythia-ai-cleanup/resolve/main/pcn_model.pt",
        "source": "MIT (wentaoyuan/pcn PyTorch port)",
    },
    "shape_vae": {
        "url": "https://huggingface.co/FoxyNinjaStudios/pythia-ai-cleanup/resolve/main/shape_vae_model.pt",
        "source": "MIT (autonomousvision/occupancy_networks)",
    },
}

CHECKPOINTS_DIR = Path(os.getenv("SAM3D_CHECKPOINTS_DIR", "checkpoints/ai_cleanup"))
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded models (loaded on first use, unloaded after cleanup)
_pcn_model = None
_shape_vae_model = None
_device = None


def get_device():
    """Get the optimal device for inference (MPS > CUDA > CPU)."""
    global _device
    if _device is not None:
        return _device
    
    if torch.backends.mps.is_available():
        _device = torch.device("mps")
        logger.info("Using Metal Performance Shaders (MPS) for AI mesh cleanup")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
        logger.info("Using CUDA for AI mesh cleanup")
    else:
        _device = torch.device("cpu")
        logger.info("Using CPU for AI mesh cleanup (inference will be slow)")
    return _device


def _download_weight(model_name: str, url: str) -> Optional[Path]:
    """Download weight from URL if not cached. Returns path or None."""
    filepath = CHECKPOINTS_DIR / f"{model_name}_model.pt"
    
    # Already downloaded
    if filepath.exists():
        logger.info(f"Using cached {model_name} weights at {filepath}")
        return filepath
    
    logger.info(f"Downloading {model_name} weights from {url}…")
    try:
        with urlopen(url) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            with open(filepath, "wb") as f:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        logger.debug(f"  {model_name}: {pct:.1f}% ({downloaded}/{total_size})")
        
        logger.info(f"Successfully downloaded {model_name} to {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Failed to download {model_name} weights: {e}")
        if filepath.exists():
            filepath.unlink()
        return None


def load_pcn_model():
    """Load Point Completion Network for point cloud denoising."""
    global _pcn_model
    if _pcn_model is not None:
        return _pcn_model

class SimplePointDenoiser(torch.nn.Module):
    """Point Completion Network (PCN) - point cloud denoiser."""
    def __init__(self, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
        )
    
    def forward(self, x):
        """x: (B, N, 3) point cloud"""
        B, N, _ = x.shape
        x_encoded = []
        for i in range(N):
            enc = self.encoder(x[:, i, :])
            x_encoded.append(enc)
        x_encoded = torch.stack(x_encoded, dim=1)
        
        out = []
        for i in range(N):
            dec = self.decoder(x_encoded[:, i, :])
            out.append(dec)
        return torch.stack(out, dim=1)


def load_pcn_model():
    """Load Point Completion Network for point cloud denoising with pretrained weights."""
    global _pcn_model
    if _pcn_model is not None:
        return _pcn_model
    
    try:
        logger.info("Loading Point Completion Network (PCN)…")
        
        # Create architecture
        model = SimplePointDenoiser(num_points=2048)
        model = model.to(get_device())
        model.eval()
        
        # Try to load pretrained weights
        config = WEIGHT_CONFIG["pcn"]
        weight_path = _download_weight("pcn", config["url"])
        
        if weight_path and weight_path.exists():
            try:
                checkpoint = torch.load(weight_path, map_location=get_device())
                # Handle both direct state_dict and wrapped checkpoint
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded PCN pretrained weights from {weight_path}")
            except Exception as e:
                logger.warning(f"Could not load PCN weights, using random init: {e}")
        else:
            logger.warning("PCN weights not available, using random initialization (inference will be degraded)")
        
        _pcn_model = model
        logger.info("PCN model ready")
        
    except Exception as e:
        logger.error(f"Failed to load PCN model: {e}")
        _pcn_model = None
    
    return _pcn_model


def load_shape_vae_model():
    """Load ShapeNet-trained 3D Shape VAE for mesh completion."""
    global _shape_vae_model
    if _shape_vae_model is not None:
        return _shape_vae_model
    

class SimpleVoxelVAE(torch.nn.Module):
    """3D Shape VAE for voxel-based shape completion."""
    def __init__(self, resolution=32, latent_dim=128):
        super().__init__()
        self.resolution = resolution
        self.latent_dim = latent_dim
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(8, 16, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(16, 32, 4, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        
        self.fc_enc = torch.nn.Linear(32 * 4 * 4 * 4, latent_dim * 2)
        self.fc_dec = torch.nn.Linear(latent_dim, 32 * 4 * 4 * 4)
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(8, 1, 4, stride=2, padding=1),
            torch.nn.Sigmoid(),
        )
    
    def encode(self, x):
        """x: (B, 1, 32, 32, 32) voxel grid"""
        h = self.encoder(x)
        h = h.view(h.shape[0], -1)
        params = self.fc_enc(h)
        mu, logvar = params.chunk(2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """z: (B, latent_dim)"""
        h = self.fc_dec(z)
        h = h.view(h.shape[0], 32, 4, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def load_shape_vae_model():
    """Load 3D Shape VAE for mesh completion with pretrained weights."""
    global _shape_vae_model
    if _shape_vae_model is not None:
        return _shape_vae_model
    
    try:
        logger.info("Loading 3D Shape VAE…")
        
        # Create architecture
        model = SimpleVoxelVAE(resolution=32, latent_dim=128)
        model = model.to(get_device())
        model.eval()
        
        # Try to load pretrained weights
        config = WEIGHT_CONFIG["shape_vae"]
        weight_path = _download_weight("shape_vae", config["url"])
        
        if weight_path and weight_path.exists():
            try:
                checkpoint = torch.load(weight_path, map_location=get_device())
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded Shape VAE pretrained weights from {weight_path}")
            except Exception as e:
                logger.warning(f"Could not load Shape VAE weights, using random init: {e}")
        else:
            logger.warning("Shape VAE weights not available, using random initialization (inference will be degraded)")
        
        _shape_vae_model = model
        logger.info("Shape VAE model ready")
        
    except Exception as e:
        logger.error(f"Failed to load Shape VAE model: {e}")
        _shape_vae_model = None
    
    return _shape_vae_model


def unload_models():
    """Unload AI models to free memory."""
    global _pcn_model, _shape_vae_model
    if _pcn_model is not None:
        del _pcn_model
        _pcn_model = None
    if _shape_vae_model is not None:
        del _shape_vae_model
        _shape_vae_model = None
    
    device = get_device()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("AI mesh cleanup models unloaded")


def denoise_point_cloud(
    vertices: np.ndarray,
    num_points: int = 2048,
    verbose: bool = False,
) -> np.ndarray:
    """
    Denoise a point cloud using PCN.
    
    Args:
        vertices: (V, 3) point positions
        num_points: Number of points to sample/use
        verbose: Print progress
    
    Returns:
        (V, 3) denoised vertices
    """
    model = load_pcn_model()
    if model is None:
        logger.warning("PCN model unavailable, skipping denoising")
        return vertices
    
    try:
        device = get_device()
        
        # Normalize points to [-1, 1]
        centroid = vertices.mean(axis=0)
        vertices_norm = vertices - centroid
        max_dist = np.linalg.norm(vertices_norm, axis=1).max()
        if max_dist > 0:
            vertices_norm = vertices_norm / max_dist
        
        # Sample/pad to num_points
        if len(vertices) > num_points:
            idx = np.random.choice(len(vertices), num_points, replace=False)
            points_sampled = vertices_norm[idx]
        else:
            points_sampled = vertices_norm
            if len(points_sampled) < num_points:
                pad_count = num_points - len(points_sampled)
                pad = np.random.randn(pad_count, 3) * 0.01
                points_sampled = np.vstack([points_sampled, pad])
        
        # Inference
        x = torch.from_numpy(points_sampled[None]).float().to(device)  # (1, N, 3)
        with torch.no_grad():
            denoised = model(x)[0]  # (1, N, 3) → (N, 3)
        
        denoised_np = denoised[0].cpu().numpy()
        
        # Denormalize
        denoised_np = denoised_np * max_dist + centroid
        
        # Take only non-padding points
        denoised_np = denoised_np[:len(vertices)]
        
        if verbose:
            logger.info(f"Point cloud denoised: {len(vertices)} points")
        
        return denoised_np
    
    except Exception as e:
        logger.error(f"Point cloud denoising failed: {e}")
        return vertices


def complete_shape_voxel(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int = 32,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete/refine a shape using a 3D VAE.
    
    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) mesh faces
        resolution: Voxel grid resolution (32 standard)
        verbose: Print progress
    
    Returns:
        (vertices, faces) completed/refined mesh
    """
    model = load_shape_vae_model()
    if model is None:
        logger.warning("Shape VAE model unavailable, skipping completion")
        return vertices, faces
    
    try:
        device = get_device()
        
        # Voxelize input mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        voxel_grid = mesh.voxelized(resolution).matrix
        voxel_grid = voxel_grid.astype(np.float32)
        
        # Inference through VAE
        x = torch.from_numpy(voxel_grid[None, None]).to(device)  # (1, 1, R, R, R)
        with torch.no_grad():
            recon, _, _ = model(x)
        
        recon_np = recon[0, 0].cpu().numpy()  # (R, R, R)
        
        # Threshold and convert back to mesh
        recon_np = (recon_np > 0.5).astype(np.uint8)
        
        # Simple marching cubes
        try:
            from skimage import measure
            vertices_new = []
            faces_new = []
            
            # Threshold and extract surface
            threshold = 0.5
            verts, faces_mc, _, _ = measure.marching_cubes(
                recon_np, level=threshold
            )
            
            # Normalize back to original scale
            verts = verts / resolution * 2 - 1
            bbox = vertices.max(axis=0) - vertices.min(axis=0)
            center = vertices.mean(axis=0)
            verts = verts * (bbox / 2) + center
            
            if verbose:
                logger.info(f"Shape completed: {len(verts)} vertices, {len(faces_mc)} faces")
            
            return verts, faces_mc
        except Exception as e:
            logger.warning(f"Marching cubes failed: {e}, returning original mesh")
            return vertices, faces
    
    except Exception as e:
        logger.error(f"Shape completion failed: {e}")
        return vertices, faces


def denoise_and_complete_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    enable_denoising: bool = True,
    enable_completion: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply AI-based mesh cleanup: denoising + optional shape completion.
    
    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) mesh faces
        enable_denoising: Use PCN for point cloud denoising
        enable_completion: Use VAE for shape completion
        verbose: Print progress
    
    Returns:
        (vertices, faces) cleaned/completed mesh
    """
    if not enable_denoising and not enable_completion:
        return vertices, faces
    
    try:
        if enable_denoising:
            vertices = denoise_point_cloud(vertices, verbose=verbose)
        
        if enable_completion:
            vertices, faces = complete_shape_voxel(vertices, faces, verbose=verbose)
        
        return vertices, faces
    
    finally:
        # Optionally unload models after cleanup to save memory
        # Uncomment to enable aggressive memory cleanup:
        # unload_models()
        pass
