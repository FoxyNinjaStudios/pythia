"""
AI-based mesh cleanup using point cloud denoising and shape completion VAE.
Runs on Metal (MPS) where available, with intelligent memory management.
Supports downloading pretrained weights from Hugging Face and Google Drive.
"""

import os
import hashlib
from pathlib import Path
from urllib.request import urlopen
from typing import Optional, Tuple
import subprocess

import numpy as np
import torch
import trimesh
import logging

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

logger = logging.getLogger(__name__)

# Weight sources (modern PyTorch models)
# Point cloud denoising: Point Transformer V3 (MIT, active development)
# Shape completion: SnowflakeNet (MIT, point cloud deconvolution with skip-transformer)
WEIGHT_CONFIG = {
    "point_transformer_v3": {
        "model": "Pointcept/PointTransformerV3",
        "pretrained": True,
        "source": "MIT (Pointcept/PointTransformerV3)",
        "paper": "https://arxiv.org/abs/2312.10017",
    },
    "snowflakenet": {
        "gdrive_folder_id": "1mdA-6ZwzXAbaWJ6fmfL9-gl3aGTGTWyR",
        "gdrive_model_file": "spd_scannet_mix.ckpt",
        "pretrained": True,
        "source": "MIT (AllenXiangX/SnowflakeNet)",
        "paper": "https://arxiv.org/abs/2202.09367",
    },
    # Fallback simple models (random init, infrastructure only)
    "simple_pcn_fallback": {
        "url": "https://huggingface.co/FoxyNinjaStudios/pythia-ai-cleanup/resolve/main/pcn_model.pt",
        "source": "Random initialization (fallback)",
    },
    "simple_vae_fallback": {
        "url": "https://huggingface.co/FoxyNinjaStudios/pythia-ai-cleanup/resolve/main/shape_vae_model.pt",
        "source": "Random initialization (fallback)",
    },
}

CHECKPOINTS_DIR = Path(os.getenv("SAM3D_CHECKPOINTS_DIR", "checkpoints/ai_cleanup"))
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded models (loaded on first use, unloaded after cleanup)
_pcn_model = None
_shape_vae_model = None
_device = None
_pcn_load_attempted = False  # Track if we've already tried to load PCN
_vae_load_attempted = False  # Track if we've already tried to load VAE


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


def _download_gdrive_weight(model_name: str, folder_id: str, filename: str) -> Optional[Path]:
    """Download weight from Google Drive folder if not cached. Returns path or None."""
    filepath = CHECKPOINTS_DIR / f"{model_name}_model.pt"
    
    # Already downloaded
    if filepath.exists():
        logger.info(f"Using cached {model_name} weights at {filepath}")
        return filepath
    
    logger.info(f"Downloading {model_name} weights from Google Drive…")
    try:
        import gdown
        import shutil
        
        # Download the entire Google Drive folder
        gdown.download_folder(folder_id, output=str(CHECKPOINTS_DIR), quiet=False)
        
        # For SnowflakeNet, look for the completion model in the downloaded folder
        if model_name == "snowflakenet":
            # The folder contains multiple model types; use the best PCN completion model
            completion_dir = CHECKPOINTS_DIR / "completion"
            if completion_dir.exists():
                # Prefer the Chamfer Distance L1 model (ckpt-best-pcn-cd_l1.pth)
                model_file = completion_dir / "ckpt-best-pcn-cd_l1.pth"
                if model_file.exists():
                    shutil.copy(str(model_file), str(filepath))
                    logger.info(f"Successfully downloaded {model_name} to {filepath}")
                    return filepath
                else:
                    logger.error(f"Downloaded folder but could not find PCN model")
                    return None
            else:
                logger.error(f"Downloaded folder but could not find completion/ subdirectory")
                return None
        else:
            # Generic lookup: look for the specified filename
            downloaded_file = CHECKPOINTS_DIR / filename
            if downloaded_file.exists():
                shutil.copy(str(downloaded_file), str(filepath))
                logger.info(f"Successfully downloaded {model_name} to {filepath}")
                return filepath
            else:
                logger.error(f"Downloaded folder but could not find {filename}")
                return None
        
    except ImportError:
        logger.error("gdown not installed; cannot download from Google Drive")
        return None
    except Exception as e:
        logger.error(f"Failed to download {model_name} from Google Drive: {e}")
        if filepath.exists():
            filepath.unlink()
        return None


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
    """Load point cloud denoising model.
    
    Point Transformer V3 is the primary choice but has compatibility issues.
    This function now returns a marker indicating Open3D fallback denoising is available.
    
    NOTE: Returns None to indicate "use Open3D denoising fallback" rather than ML-based denoising.
    The actual denoising happens in denoise_point_cloud_open3d().
    """
    global _pcn_model, _pcn_load_attempted
    
    if _pcn_load_attempted:
        return _pcn_model  # Return cached result
    
    _pcn_load_attempted = True
    
    if not OPEN3D_AVAILABLE:
        logger.warning("⚠ Point cloud denoising is disabled: Open3D not available")
        logger.warning("  Install open3d to enable robust statistical denoising")
        _pcn_model = None
        return None
    
    logger.info("✓ Point cloud denoising: Using Open3D statistical denoising (fallback for Point Transformer V3)")
    logger.info("  - Statistical outlier removal + radius-based filtering")
    logger.info("  - Deterministic, no ML weights needed")
    
    _pcn_model = "open3d_fallback"  # Marker indicating to use Open3D denoising
    return _pcn_model




def load_shape_vae_model():
    """Load shape completion model.
    
    Attempts to load SnowflakeNet (modern point cloud deconvolution with skip-transformer).
    Falls back to SimpleVoxelVAE if SnowflakeNet unavailable.
    """
    global _shape_vae_model, _vae_load_attempted
    
    if _vae_load_attempted:
        return _shape_vae_model  # Return cached result
    
    _vae_load_attempted = True
    device = get_device()
    snowflakenet_path = CHECKPOINTS_DIR / "snowflakenet_model.pt"
    
    # Try SnowflakeNet first if weights exist
    if snowflakenet_path.exists():
        try:
            logger.info(f"Loading SnowflakeNet from {snowflakenet_path}…")
            checkpoint = torch.load(snowflakenet_path, map_location=device)
            
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                logger.info("✓ SnowflakeNet checkpoint loaded successfully")
                logger.info("  SnowflakeNet: Point cloud deconvolution with skip-transformer (ICCV 2021, TPAMI 2023)")
                _shape_vae_model = checkpoint
                return _shape_vae_model
        except Exception as e:
            logger.warning(f"Could not load SnowflakeNet checkpoint: {e}")
            logger.info("Falling back to SimpleVoxelVAE…")
    else:
        logger.info("SnowflakeNet checkpoint not found, using SimpleVoxelVAE fallback…")
    
    # Fallback to SimpleVoxelVAE
    try:
        logger.info("Loading simple voxel VAE for shape completion…")
        
        model = SimpleVoxelVAE(resolution=32, latent_dim=128)
        model = model.to(device)
        model.eval()
        
        # Try to load fallback weights if available
        config = WEIGHT_CONFIG["simple_vae_fallback"]
        weight_path = _download_weight("vae_fallback", config["url"])
        
        if weight_path and weight_path.exists():
            try:
                ckpt = torch.load(weight_path, map_location=device)
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    model.load_state_dict(ckpt["state_dict"])
                else:
                    model.load_state_dict(ckpt)
                logger.info(f"Loaded Shape VAE weights from {weight_path}")
            except Exception as e:
                logger.warning(f"Could not load Shape VAE weights, using random init: {e}")
        else:
            logger.warning("Shape VAE weights not available, using random initialization")
        
        _shape_vae_model = model
        logger.info("✓ Shape VAE ready for shape completion (random initialization)")
        
    except Exception as e:
        logger.error(f"Failed to load Shape VAE model: {e}")
        _shape_vae_model = None
    
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


def denoise_point_cloud_open3d(
    vertices: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    radius: float = None,
    min_points: int = 10,
    apply_smooth: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Denoise a point cloud using Open3D statistical methods (fallback for Point Transformer V3).
    
    This combines two proven techniques:
    1. Statistical Outlier Removal (SOR): Removes points farther than std_ratio * std_dev from neighbors
    2. Radius-based Outlier Removal (ROR): Removes points with too few neighbors in a radius
    
    Args:
        vertices: (V, 3) point positions
        nb_neighbors: Number of neighbors to consider for statistics (default 20)
        std_ratio: Standard deviation multiplier for outlier threshold (default 2.0, stricter = higher)
        radius: Radius for neighbor search (auto-estimated from point density if None)
        min_points: Minimum neighbors within radius (ROR threshold)
        apply_smooth: If True, apply bilateral filtering after outlier removal
        verbose: Print progress
    
    Returns:
        (V', 3) denoised vertices (may be fewer points if outliers removed)
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available, returning original vertices")
        return vertices
    
    if verbose:
        logger.info(f"Denoising point cloud: {len(vertices)} points → Open3D statistical methods")
    
    try:
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        
        # Estimate radius from average point spacing if not provided
        if radius is None:
            # Rough estimate: radius ~ 1.5x average nearest neighbor distance
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
            dists = []
            kdtree = o3d.geometry.KDTreeFlann(pcd)
            for i in range(min(1000, len(vertices))):  # Sample to estimate
                _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], 2)
                if len(idx) > 1:
                    dist = np.linalg.norm(np.asarray(pcd.points[idx[1]]) - np.asarray(pcd.points[i]))
                    dists.append(dist)
            avg_spacing = np.median(dists) if dists else 0.01
            radius = avg_spacing * 3.0  # 3x nearest neighbor distance
        
        # Statistical Outlier Removal
        pcd_sor, idx_sor = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        if verbose:
            logger.info(f"  After SOR: {len(pcd_sor.points)} points (removed {len(vertices) - len(pcd_sor.points)})")
        
        # Radius-based Outlier Removal
        pcd_ror, idx_ror = pcd_sor.remove_radius_outlier(
            nb_points=min_points,
            radius=radius
        )
        
        if verbose:
            logger.info(f"  After ROR: {len(pcd_ror.points)} points (removed {len(pcd_sor.points) - len(pcd_ror.points)})")
        
        # Optional bilateral filtering for additional smoothing
        if apply_smooth and len(pcd_ror.points) > 0:
            try:
                # Try Laplacian smoothing (available in newer versions)
                pcd_ror = pcd_ror.filter_smooth_laplacian(
                    number_of_iterations=5,
                    lambda_filter=0.5
                )
                if verbose:
                    logger.info(f"  After smoothing: {len(pcd_ror.points)} points")
            except AttributeError:
                # Fallback: use simple averaging if Laplacian smoothing unavailable
                if verbose:
                    logger.info(f"  Laplacian smoothing unavailable, skipping")

        
        result = np.asarray(pcd_ror.points, dtype=np.float32)
        
        if verbose:
            logger.info(f"Denoising complete: {len(result)} points")
        
        return result
    
    except Exception as e:
        logger.error(f"Open3D denoising failed: {e}, returning original vertices")
        return vertices
def denoise_point_cloud(
    vertices: np.ndarray,
    num_points: int = 2048,
    verbose: bool = False,
) -> np.ndarray:
    """
    Denoise a point cloud using available denoising method.
    
    Tries ML-based Point Transformer V3 first, falls back to Open3D statistical denoising.
    
    Args:
        vertices: (V, 3) point positions
        num_points: Number of points to sample/use (for ML methods)
        verbose: Print progress
    
    Returns:
        (V, 3) or (V', 3) denoised vertices
    """
    model = load_pcn_model()
    
    # Use Open3D fallback if available
    if model == "open3d_fallback":
        return denoise_point_cloud_open3d(
            vertices,
            nb_neighbors=20,
            std_ratio=2.0,
            apply_smooth=True,
            verbose=verbose
        )
    
    # Model is None or unavailable
    if model is None:
        logger.warning("No denoising method available (Open3D not installed), returning original vertices")
        return vertices
    
    # ML-based denoising (if Point Transformer V3 becomes available)
    try:
        device = get_device()
        original_count = len(vertices)
        
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
            output = model(x)
            # Handle different output formats (some models return tensor directly, others return tuple)
            if isinstance(output, tuple):
                denoised = output[0]
            else:
                denoised = output
            
            # Ensure we have the right shape
            if denoised.ndim == 3:  # (B, N, 3)
                denoised = denoised[0]  # Take first batch element: (N, 3)
        
        denoised_np = denoised.cpu().numpy()
        
        # Denormalize
        denoised_np = denoised_np * max_dist + centroid
        
        # Take only original point count (remove padding)
        denoised_np = denoised_np[:original_count]
        
        if verbose:
            logger.info(f"Point cloud denoised (ML): {len(vertices)} points")
        
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
            output = model(x)
            if isinstance(output, tuple):
                recon = output[0]  # (B, 1, R, R, R)
            else:
                recon = output
        
        recon_np = recon[0, 0].cpu().numpy()  # (R, R, R)
        
        # Threshold and convert back to mesh
        recon_np = (recon_np > 0.5).astype(np.uint8)
        
        # Simple marching cubes to extract surface
        try:
            from skimage import measure
            
            # Extract isosurface using marching cubes
            verts, faces_mc, _, _ = measure.marching_cubes(
                recon_np, level=0.5
            )
            
            if len(verts) == 0:
                logger.warning("Marching cubes produced empty mesh, returning original")
                return vertices, faces
            
            # Normalize verts back to original coordinate frame
            # Voxel coords [0, R] -> normalize to [-1, 1] -> scale to original bbox
            verts = verts / resolution * 2 - 1
            
            # Get original mesh bounds
            center = vertices.mean(axis=0)
            bbox = vertices.max(axis=0) - vertices.min(axis=0)
            max_extent = np.max(bbox)
            
            # Scale verts to match original extent
            verts = verts * (max_extent / 2) + center
            
            if verbose:
                logger.info(f"Shape completed: {len(verts)} vertices, {len(faces_mc)} faces (from {len(vertices)} original)")
            
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
