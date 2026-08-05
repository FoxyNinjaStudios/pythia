# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Low-memory inference pipeline for SAM-3D.

This pipeline loads models sequentially and deletes them after use,
reducing peak memory from ~45GB to ~15GB. Stage 2 (SLAT) can optionally
run on MPS (Metal GPU) for improved speed on Apple Silicon.
"""

import os
import gc
import time
from typing import Union, Optional
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d

from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.pipeline.inference_pipeline_pointmap import (
    InferencePipelinePointMap,
    camera_to_pytorch3d_camera,
)
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import get_mask
from sam3d_objects.data.dataset.tdfy.transforms_3d import DecomposedTransform
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
from sam3d_objects.pipeline.inference_utils import (
    get_pose_decoder,
    SLAT_MEAN,
    SLAT_STD,
    downsample_sparse_structure,
    prune_sparse_structure,
)
from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY & DEVICE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def force_gc():
    """Aggressive garbage collection and cache clearing across all devices."""
    gc.collect()
    gc.collect()
    gc.collect()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception as e:
            logger.debug(f"[MEM] MPS cache clear failed: {e}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def delete_model_completely(model, name="model"):
    """Fully delete a model and all its parameters from memory."""
    if model is None:
        return
    
    try:
        # Move to CPU first to release device memory
        model.cpu()
        
        # Delete all parameters and buffers
        for param in model.parameters():
            param.data = torch.empty(0)
            if param.grad is not None:
                param.grad = None
        
        for buffer_name, buffer in list(model.named_buffers()):
            buffer.data = torch.empty(0)
        
        # Clear module dict
        if hasattr(model, '_modules'):
            model._modules.clear()
        
        del model
        logger.debug(f"[MEM] Deleted {name}")
    except Exception as e:
        logger.warning(f"[MEM] Failed to delete {name}: {e}")
    
    force_gc()


def get_memory_gb():
    """Get current process memory usage in GB (macOS peak RSS)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 ** 3)
    except Exception:
        return -1


def log_memory(stage: str):
    """Log current memory usage with stage label."""
    mem = get_memory_gb()
    if mem > 0:
        logger.info(f"[MEM] {stage}: {mem:.1f} GB")


def get_stage2_device(use_mps: bool) -> torch.device:
    """Determine Stage 2 execution device (MPS or CPU)."""
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-VIEW FUSION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_view_confidence_weights(results, fusion_config=None):
    """
    Compute per-view confidence weights based on surface properties and reconstruction quality.
    
    Higher confidence for views with:
    - Lower coordinate variance (more consistent reconstruction)
    - Tighter geometry (higher precision)
    
    Args:
        results: List of per-view reconstruction results
        fusion_config: Optional config (dict or object) with weighting strategy
    
    Returns:
        List of normalized confidence weights [0.0-1.0] per view
    """
    if not results or len(results) < 2:
        return [1.0] * len(results) if results else []
    
    if fusion_config is None:
        fusion_config = {}
    
    # Helper to get values from dict or object
    def get_config_value(cfg, key, default):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        else:
            return getattr(cfg, key, default)
    
    weighting_mode = get_config_value(fusion_config, "view_weighting", "uniform")
    logger.info(f"[MV] Computing view confidence weights ({weighting_mode} mode)")
    
    weights = []
    
    if weighting_mode == "uniform":
        # Equal weighting for all views
        weights = [1.0 / len(results)] * len(results)
    
    elif weighting_mode == "entropy":
        # Weight by reconstruction clarity (lower entropy/variance = higher confidence)
        uncertainties = []
        for i, result in enumerate(results):
            try:
                # Estimate uncertainty from coordinate distribution
                if "coords" in result and result["coords"] is not None:
                    coords = torch.as_tensor(result["coords"], dtype=torch.float32)
                    if coords.numel() > 0:
                        # Standard deviation of coordinates as uncertainty measure
                        uncertainty = float(coords.std().item())
                        uncertainties.append(uncertainty)
                    else:
                        uncertainties.append(1.0)
                else:
                    uncertainties.append(1.0)
            except Exception as e:
                logger.debug(f"[MV] Uncertainty computation failed for view {i}: {e}")
                uncertainties.append(1.0)
        
        # Invert: lower uncertainty -> higher weight
        if uncertainties and max(uncertainties) > 0:
            max_unc = max(uncertainties)
            min_unc = min(uncertainties)
            unc_range = max_unc - min_unc + 1e-8
            weights = [(max_unc - u) / unc_range for u in uncertainties]
        else:
            weights = [1.0 / len(results)] * len(results)
    
    else:
        # Default to uniform
        weights = [1.0 / len(results)] * len(results)
    
    # Normalize weights to sum to 1.0
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        weights = [1.0 / len(results)] * len(results)
    
    logger.info(f"[MV] View confidence weights: {[f'{w:.3f}' for w in weights]}")
    return weights


def fuse_gaussians_with_confidence(gaussian_list, confidence_weights=None):
    """
    Fuse multiple 3D gaussians with confidence-weighted averaging.
    
    Averages gaussian positions, colors, and covariances weighted by per-view confidence.
    
    Args:
        gaussian_list: List of gaussian objects from each view
        confidence_weights: Optional list of confidence weights (sum to 1.0)
    
    Returns:
        Fused gaussian object with blended appearance
    """
    if not gaussian_list or len(gaussian_list) == 0:
        return None
    
    if len(gaussian_list) == 1:
        return gaussian_list[0]
    
    try:
        if confidence_weights is None:
            confidence_weights = [1.0 / len(gaussian_list)] * len(gaussian_list)
        
        # Extract gaussian data
        xyz_list = []
        rgb_list = []
        opacities_list = []
        
        for i, gaussian in enumerate(gaussian_list):
            if gaussian is None:
                continue
            try:
                xyz_list.append(gaussian.get_xyz)
                rgb_list.append(gaussian.get_features[:, :3])
                opacities_list.append(gaussian.get_opacity)
            except Exception as e:
                logger.debug(f"[MV] Failed to extract gaussian data from view {i}: {e}")
                continue
        
        if not xyz_list:
            logger.warning("[MV] No valid gaussians to fuse")
            return gaussian_list[0] if gaussian_list[0] is not None else None
        
        # Weighted fusion
        fused_xyz = torch.zeros_like(xyz_list[0])
        fused_rgb = torch.zeros_like(rgb_list[0])
        fused_opacity = torch.zeros_like(opacities_list[0])
        
        for i, (xyz, rgb, opacity) in enumerate(zip(xyz_list, rgb_list, opacities_list)):
            w = confidence_weights[i % len(confidence_weights)]
            fused_xyz = fused_xyz + xyz * w
            fused_rgb = fused_rgb + rgb * w
            fused_opacity = fused_opacity + opacity * w
        
        # Update base gaussian with fused data
        fused = gaussian_list[0]
        if hasattr(fused, '_xyz'):
            fused._xyz = fused_xyz
        if hasattr(fused, '_features_dc'):
            fused._features_dc = fused_rgb.unsqueeze(1) if fused_rgb.dim() == 2 else fused_rgb
        if hasattr(fused, '_opacity'):
            fused._opacity = fused_opacity
        
        logger.info(f"[MV] Fused {len(xyz_list)} gaussians with confidence weights")
        return fused
    
    except Exception as e:
        logger.warning(f"[MV] Gaussian fusion failed ({e}); using first view")
        return gaussian_list[0] if gaussian_list else None


# ─────────────────────────────────────────────────────────────────────────────
# CACHING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save_cache(data: dict, cache_path: str):
    """Save intermediate outputs to cache file."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(data, cache_path)
        logger.info(f"[CACHE] Saved to {cache_path}")
    except Exception as e:
        logger.warning(f"[CACHE] Failed to save: {e}")


def load_cache(cache_path: str) -> Optional[dict]:
    """Load intermediate outputs from cache file."""
    try:
        if os.path.exists(cache_path):
            logger.info(f"[CACHE] Loading from {cache_path}")
            return torch.load(cache_path, weights_only=False)
    except Exception as e:
        logger.warning(f"[CACHE] Failed to load: {e}")
    return None


def get_cache_path(cache_dir: str, stage: str, input_hash: str) -> str:
    """Get cache file path for a given stage."""
    return os.path.join(cache_dir, f"{stage}_{input_hash}.pt")


def compute_input_hash(image: np.ndarray, mask: np.ndarray) -> str:
    """Compute a hash of the input for cache key."""
    import hashlib
    # Simple hash based on image shape and a sample of pixel values
    data = f"{image.shape}_{mask.shape}_{image.mean():.4f}_{mask.sum()}"
    return hashlib.md5(data.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class InferencePipelineLowMemory:
    """
    Low-memory version of InferencePipelinePointMap.
    
    Models are loaded on-demand and deleted after each stage, reducing peak
    memory from ~45GB to ~15GB. Stage 2 (SLAT) can optionally run on MPS for
    improved speed on Apple Silicon.
    """
    
    def __init__(
        self,
        config_path: str,
        depth_model=None,
        clip_pointmap_beyond_scale=None,
        device="cpu",
        dtype="float16",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the low-memory pipeline.
        
        Models are NOT loaded here; they are loaded on-demand during run().
        
        Args:
            config_path: Path to pipeline configuration (pipeline.yaml)
            depth_model: Pre-loaded depth model (optional, loaded on-demand if None)
            device: Base device for model loading ("cpu" or "mps")
            dtype: Model precision ("float16", "bfloat16", "float32")
            cache_dir: Optional directory for caching intermediate SLAT outputs
        """
        self.config_path = config_path
        self.workspace_dir = os.path.dirname(config_path)
        self.config = OmegaConf.load(config_path)
        self.device = torch.device(device)
        self.dtype = self._get_dtype(dtype)
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        self.cache_dir = cache_dir
        
        # Pipeline settings from config
        self.decode_formats = self.config.get("decode_formats", ["mesh"])
        self.pad_size = self.config.get("pad_size", 1.0)
        self.version = self.config.get("version", "v0")
        self.downsample_ss_dist = self.config.get("downsample_ss_dist", 0)
        self.ss_max_coords = self.config.get("ss_max_coords", 42000)
        self.ss_full_res_max_coords = int(
            os.environ.get(
                "SAM3D_FULL_RES_MAX_COORDS",
                self.config.get("ss_full_res_max_coords", 30000),
            )
        )
        
        # Inference steps and configuration
        self.ss_inference_steps = self.config.get("ss_inference_steps", 25)
        self.ss_rescale_t = self.config.get("ss_rescale_t", 3)
        self.ss_cfg_strength = self.config.get("ss_cfg_strength", 7)
        self.ss_cfg_interval = self.config.get("ss_cfg_interval", [0, 500])
        self.ss_cfg_strength_pm = self.config.get("ss_cfg_strength_pm", 0.0)
        
        self.slat_inference_steps = self.config.get("slat_inference_steps", 25)
        self.slat_rescale_t = self.config.get("slat_rescale_t", 3)
        self.slat_cfg_strength = self.config.get("slat_cfg_strength", 5)
        self.slat_cfg_interval = self.config.get("slat_cfg_interval", [0, 500])
        self.slat_mean = torch.tensor(self.config.get("slat_mean", SLAT_MEAN))
        self.slat_std = torch.tensor(self.config.get("slat_std", SLAT_STD))
        
        # Initialize lightweight components (remain in memory)
        self.ss_preprocessor = instantiate(self.config.ss_preprocessor)
        self.slat_preprocessor = instantiate(self.config.slat_preprocessor)
        self.pose_decoder = get_pose_decoder(self.config.get("pose_decoder_name", "default"))
        
        # Depth model: can be pre-loaded or loaded on-demand
        self.depth_model_config = self.config.get("depth_model", None)
        self.depth_model = depth_model
        
        if cache_dir:
            logger.info(f"[INIT] Pipeline initialized with cache at: {cache_dir}")
        else:
            logger.info(f"[INIT] Pipeline initialized (no caching)")
        log_memory("After init")
    
    @staticmethod
    def _get_dtype(dtype: str) -> torch.dtype:
        """Map string dtype to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")
        return dtype_map[dtype]
    
    def _load_model(self, config_key: str, ckpt_key: str, state_dict_fn=None):
        """Load a model from checkpoint."""
        config_path = os.path.join(self.workspace_dir, self.config[config_key])
        ckpt_path = os.path.join(self.workspace_dir, self.config[ckpt_key])
        
        logger.info(f"[LOAD] Loading {config_key}...")
        config = OmegaConf.load(config_path)
        
        # Remove pretrained path if present
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        
        model = instantiate(config)
        
        # Load checkpoint
        try:
            if ckpt_path.endswith(".safetensors"):
                state_dict = load_file(ckpt_path, device="cpu")
                if state_dict_fn is not None:
                    state_dict = state_dict_fn(state_dict)
                model.load_state_dict(state_dict, strict=False)
            else:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_dict = checkpoint.get("state_dict", checkpoint)
                if state_dict_fn is not None:
                    state_dict = state_dict_fn(state_dict)
                model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.error(f"[LOAD] Failed to load {config_key}: {e}")
            raise
        
        model = model.to(self.device)
        model.eval()
        log_memory(f"After loading {config_key}")
        return model
    
    def _load_generator(self, config_key: str, ckpt_key: str):
        """Load a generator model with condition embedder."""
        config_path = os.path.join(self.workspace_dir, self.config[config_key])
        ckpt_path = os.path.join(self.workspace_dir, self.config[ckpt_key])
        
        logger.info(f"[LOAD] Loading {config_key} with embedder...")
        full_config = OmegaConf.load(config_path)
        
        # Load generator
        gen_config = full_config["module"]["generator"]["backbone"]
        state_dict_fn = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")
        generator = instantiate(gen_config)
        
        try:
            if ckpt_path.endswith(".safetensors"):
                state_dict = load_file(ckpt_path, device="cpu")
            else:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_dict = checkpoint.get("state_dict", checkpoint)
            
            gen_state_dict = state_dict_fn(state_dict.copy())
            generator.load_state_dict(gen_state_dict, strict=False)
        except Exception as e:
            logger.error(f"[LOAD] Failed to load generator: {e}")
            raise
        
        generator = generator.to(self.device)
        generator.eval()
        
        # Load condition embedder if present
        cond_embedder = None
        if "condition_embedder" in full_config["module"]:
            try:
                cond_config = full_config["module"]["condition_embedder"]["backbone"]
                cond_state_dict_fn = filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder.")
                cond_embedder = instantiate(cond_config)
                cond_state_dict = cond_state_dict_fn(state_dict.copy())
                cond_embedder.load_state_dict(cond_state_dict, strict=False)
                cond_embedder = cond_embedder.to(self.device)
                cond_embedder.eval()
            except Exception as e:
                logger.warning(f"[LOAD] Failed to load condition embedder: {e}")
        
        log_memory(f"After loading {config_key}")
        return generator, cond_embedder
    
    def _load_depth_model(self):
        """Load depth model (MoGe) on demand."""
        if self.depth_model is not None:
            return self.depth_model
        
        if self.depth_model_config is None:
            raise ValueError("No depth model config provided")
        
        logger.info("[LOAD] Loading depth model (MoGe)...")
        self.depth_model = instantiate(self.depth_model_config)
        
        # MoGe runs on Metal when SAM3D_MOGE_DEVICE=mps (set by launcher)
        if os.environ.get("SAM3D_MOGE_DEVICE") == "mps" and hasattr(self.depth_model, 'model'):
            logger.info("[LOAD] Moving depth model to MPS")
            self.depth_model.device = torch.device("mps")
            self.depth_model.model.to("mps")
        
        log_memory("After loading depth model")
        return self.depth_model
    
    def _unload_depth_model(self):
        """Unload the depth model to free memory."""
        if self.depth_model is not None:
            if hasattr(self.depth_model, 'model'):
                delete_model_completely(self.depth_model.model, "depth_model.model")
            self.depth_model = None
            force_gc()
            log_memory("After unloading depth model")
    
    def image_to_float(self, image) -> np.ndarray:
        """Convert image to float32 normalized to [0, 1]."""
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        return image
    
    def merge_image_and_mask(self, image, mask) -> np.ndarray:
        """Merge RGB image and mask into RGBA format."""
        image = np.array(image) if isinstance(image, Image.Image) else np.array(image)
        
        if mask is not None:
            mask = np.array(mask)
            
            # Normalize mask to uint8 (0-255)
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            elif mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            
            # Add alpha channel if needed
            if mask.ndim == 2:
                mask = mask[..., None]
            
            # Combine RGB with mask as alpha
            image = np.concatenate([image[..., :3], mask], axis=-1)
        
        return image
    
    def compute_pointmap(self, image) -> dict:
        """Compute depth and pointmap using MoGe."""
        loaded_image = self.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_mask = loaded_image[..., -1]
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]
        
        depth_model = self._load_depth_model()
        
        with torch.no_grad():
            with torch.inference_mode():
                output = depth_model(loaded_image)
        
        # Move pointmaps to CPU (pytorch3d Transform3d doesn't support MPS)
        pointmaps = output["pointmaps"].float().cpu()
        camera_convention_transform = (
            Transform3d()
            .rotate(camera_to_pytorch3d_camera(device="cpu").rotation)
            .to("cpu")
        )
        points_tensor = camera_convention_transform.transform_points(pointmaps)
        intrinsics = output.get("intrinsics", None)
        if intrinsics is not None and hasattr(intrinsics, 'cpu'):
            intrinsics = intrinsics.cpu()
        
        points_tensor = points_tensor.permute(2, 0, 1)
        
        point_map_tensor = {
            "pointmap": points_tensor,
            "pts_color": loaded_image.cpu(),
        }
        
        if intrinsics is None:
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor.permute(1, 2, 0), device="cpu"
            )
            point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]
        else:
            point_map_tensor["intrinsics"] = intrinsics
        
        return point_map_tensor
    
    def preprocess_image(self, image, preprocessor, pointmap=None) -> dict:
        """Preprocess image for model input."""
        image = np.array(image) if not isinstance(image, np.ndarray) else image
        
        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")
        
        preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap
        )
        
        item = {
            "mask": preprocessor_return_dict["mask"][None].to(self.device),
            "image": preprocessor_return_dict["image"][None].to(self.device),
            "rgb_image": preprocessor_return_dict["rgb_image"][None].to(self.device),
            "rgb_image_mask": preprocessor_return_dict["rgb_image_mask"][None].to(self.device),
        }
        
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = preprocessor_return_dict["pointmap"][None].to(self.device)
            item["rgb_pointmap"] = preprocessor_return_dict["rgb_pointmap"][None].to(self.device)
            item["pointmap_scale"] = preprocessor_return_dict["pointmap_scale"][None].to(self.device)
            item["pointmap_shift"] = preprocessor_return_dict["pointmap_shift"][None].to(self.device)
            item["rgb_pointmap_scale"] = preprocessor_return_dict["rgb_pointmap_scale"][None].to(self.device)
            item["rgb_pointmap_shift"] = preprocessor_return_dict["rgb_pointmap_shift"][None].to(self.device)
        
        return item
    
    @staticmethod
    def _down_sample_img(img_3chw: torch.Tensor) -> torch.Tensor:
        """Downsample image by adaptive factor based on size."""
        x = img_3chw.unsqueeze(0)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        max_side = max(x.shape[2], x.shape[3])
        if max_side > 3800:
            scale_factor = 0.125
        elif max_side > 1900:
            scale_factor = 0.25
        elif max_side > 1200:
            scale_factor = 0.5
        else:
            scale_factor = 1.0
        
        if scale_factor < 1.0:
            x = torch.nn.functional.interpolate(
                x, scale_factor=(scale_factor, scale_factor),
                mode="bilinear", align_corners=False, antialias=True,
            )
        
        return x.squeeze(0)
    
    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only: bool = False,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_vertex_color: bool = True,
        stage1_inference_steps: Optional[int] = None,
        stage2_inference_steps: Optional[int] = None,
        use_stage1_distillation: bool = False,
        use_stage2_distillation: bool = False,
        use_stage2_mps: bool = True,  # MPS enabled by default
        decode_formats: Optional[list] = None,
        use_cache: bool = True,
        simplify_ratio: float = 0.0,
        load_slat: Optional[str] = None,
        texture_bake: bool = False,
        texture_bake_source: str = "gaussian",
        texture_size: int = 2048,
        vertex_color_source: str = "gaussian",
        full_res_geometry: bool = True,  # Keep native 64³ by default
    ) -> dict:
        """
        Run the full inference pipeline with sequential model loading.
        
        Each model is loaded, used, then deleted to minimize peak memory.
        Stage 2 (SLAT) can optionally run on MPS for GPU acceleration.
        """
        logger.info("[PIPE] Starting inference pipeline")
        log_memory("Pipeline start")
        
        # Timing checkpoints
        _stage_secs = {}
        _t_mark = {"t": time.perf_counter()}
        
        def _ck(label: str):
            """Checkpoint: record elapsed time since last checkpoint."""
            now = time.perf_counter()
            _stage_secs[label] = _stage_secs.get(label, 0.0) + (now - _t_mark["t"])
            _t_mark["t"] = now
        
        # Merge image and mask
        image = self.merge_image_and_mask(image, mask)
        
        # Initialize return values and cache state
        input_hash = None
        slat_from_cache = False
        slat = None
        ss_return_dict = {}
        pts = None
        pts_colors = None
        intrinsics = None
        
        # ────────────────────────────────────────────────────────────────────────
        # SLAT CACHE LOADING (skip stages 0, 1, 2 if available)
        # ────────────────────────────────────────────────────────────────────────
        
        if load_slat:
            logger.info(f"[CACHE] Loading SLAT from: {load_slat}")
            try:
                slat_cache = torch.load(load_slat, map_location="cpu")
                slat = sp.SparseTensor(
                    coords=slat_cache["coords"].to(self.device),
                    feats=slat_cache["feats"].to(self.device)
                )
                ss_return_dict.update({
                    "translation": slat_cache.get("ss_return_dict_translation"),
                    "scale": slat_cache.get("ss_return_dict_scale"),
                    "coords": slat_cache.get("coords"),
                    "rotation": slat_cache.get("ss_return_dict_rotation"),
                })
                intrinsics = slat_cache.get("intrinsics")
                pts = slat_cache.get("pts")
                pts_colors = slat_cache.get("pts_colors")
                slat_from_cache = True
                logger.info(f"[CACHE] Loaded SLAT with {slat.coords.shape[0]} voxels - skipping Stages 0, 1, 2")
            except Exception as e:
                logger.error(f"[CACHE] Failed to load SLAT: {e}")
                raise
        
        elif self.cache_dir and use_cache:
            np_image = np.array(image) if isinstance(image, Image.Image) else image
            np_mask = np.array(mask) if isinstance(mask, (Image.Image, np.ndarray)) else np.zeros((1,))
            input_hash = compute_input_hash(np_image, np_mask)
            
            slat_cache_path = get_cache_path(self.cache_dir, "stage2_slat", input_hash)
            slat_cache = load_cache(slat_cache_path)
            
            if slat_cache is not None:
                try:
                    slat = sp.SparseTensor(
                        coords=slat_cache["coords"].to(self.device),
                        feats=slat_cache["feats"].to(self.device)
                    )
                    ss_return_dict.update({
                        "translation": slat_cache.get("ss_return_dict_translation"),
                        "scale": slat_cache.get("ss_return_dict_scale"),
                        "coords": slat_cache.get("coords"),
                        "rotation": slat_cache.get("ss_return_dict_rotation"),
                    })
                    intrinsics = slat_cache.get("intrinsics")
                    pts = slat_cache.get("pts")
                    pts_colors = slat_cache.get("pts_colors")
                    slat_from_cache = True
                    logger.info("[CACHE] Found cached SLAT - skipping Stages 0, 1, 2")
                except Exception as e:
                    logger.warning(f"[CACHE] Failed to load cached SLAT: {e}")
        
        # ────────────────────────────────────────────────────────────────────────
        # STAGE 0: DEPTH ESTIMATION
        # ────────────────────────────────────────────────────────────────────────
        
        if not slat_from_cache:
            _ck("setup")
            logger.info("[S0] === STAGE 0: Depth Estimation ===")
            
            pointmap_dict = self.compute_pointmap(image)
            pointmap = pointmap_dict["pointmap"]
            pts = self._down_sample_img(pointmap)
            pts_colors = self._down_sample_img(pointmap_dict["pts_color"])
            intrinsics = pointmap_dict.get("intrinsics")
            
            self._unload_depth_model()
            log_memory("After depth stage")
            
            # Preprocess for Stage 1 and Stage 2
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor, pointmap=pointmap)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            
            if seed is not None:
                torch.manual_seed(seed)
            
            # ────────────────────────────────────────────────────────────────────
            # STAGE 1: SPARSE STRUCTURE
            # ────────────────────────────────────────────────────────────────────
            
            _ck("depth")
            logger.info("[S1] === STAGE 1: Sparse Structure Generation ===")
            
            ss_generator, ss_condition_embedder = self._load_generator(
                "ss_generator_config_path", "ss_generator_ckpt_path"
            )
            ss_decoder = self._load_model("ss_decoder_config_path", "ss_decoder_ckpt_path")
            
            # Configure Stage 1 generator
            ss_generator.inference_steps = stage1_inference_steps or self.ss_inference_steps
            ss_generator.reverse_fn.interval = self.ss_cfg_interval
            ss_generator.rescale_t = self.ss_rescale_t
            ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
            ss_generator.reverse_fn.unconditional_handling = "add_flag"
            
            if use_stage1_distillation:
                ss_generator.no_shortcut = False
                ss_generator.reverse_fn.strength = 0
                ss_generator.reverse_fn.strength_pm = 0
            else:
                ss_generator.no_shortcut = True
                ss_generator.reverse_fn.strength = self.ss_cfg_strength
                ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm
            
            # Run Stage 1
            with torch.no_grad():
                with torch.inference_mode():
                    bs = ss_input_dict["image"].shape[0]
                    
                    # Determine latent shape
                    if hasattr(ss_generator.reverse_fn.backbone, "latent_mapping"):
                        latent_shape_dict = {
                            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                            for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                        }
                    else:
                        latent_shape_dict = (bs,) + (4096, 8)
                    
                    # Get condition embeddings
                    if ss_condition_embedder is not None:
                        cond_tokens = ss_condition_embedder(**ss_input_dict)
                        condition_args = (cond_tokens,)
                        condition_kwargs = {}
                    else:
                        condition_args = ()
                        condition_kwargs = ss_input_dict
                    
                    # Generate latent
                    return_dict = ss_generator(
                        latent_shape_dict,
                        ss_input_dict["image"].device,
                        *condition_args,
                        **condition_kwargs,
                    )
                    
                    if not hasattr(ss_generator.reverse_fn.backbone, "latent_mapping"):
                        return_dict = {"shape": return_dict}
                    
                    # Decode to occupancy grid
                    shape_latent = return_dict["shape"]
                    ss = ss_decoder(
                        shape_latent.permute(0, 2, 1).contiguous()
                        .view(shape_latent.shape[0], 8, 16, 16, 16)
                    )
                    coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
                    
                    return_dict["coords_original"] = coords
                    original_shape = coords.shape
                    
                    # Handle resolution and downsampling
                    _full_res = full_res_geometry or os.environ.get(
                        "SAM3D_FULL_RES_GEOMETRY", "1"
                    ) != "0"
                    _force_downsample = os.environ.get(
                        "SAM3D_FORCE_DOWNSAMPLE", "1"
                    ) != "0"
                    _prune_dist = max(self.downsample_ss_dist, 1) if _full_res else self.downsample_ss_dist
                    
                    if _prune_dist > 0:
                        coords = prune_sparse_structure(coords, max_neighbor_axes_dist=_prune_dist)
                    
                    _n_surface = coords.shape[0]
                    _guard = (
                        min(self.ss_full_res_max_coords, self.ss_max_coords)
                        if _full_res
                        else self.ss_max_coords
                    )
                    coords, downsample_factor = downsample_sparse_structure(
                        coords,
                        max_coords=_guard,
                        force=_force_downsample,
                        subsample_cap=self.ss_max_coords,
                    )
                    
                    if downsample_factor > 1:
                        logger.info(
                            f"[S1] Surface {_n_surface} voxels → downsampled "
                            f"(factor={downsample_factor}, full_res={_full_res})"
                        )
                    
                    logger.info(
                        f"[S1] Coords: {original_shape[0]} → {coords.shape[0]} "
                        f"(prune_dist={_prune_dist}, downsample_factor={downsample_factor})"
                    )
                    
                    return_dict["coords"] = coords
                    return_dict["downsample_factor"] = downsample_factor
            
            # Run pose decoder
            pointmap_scale = ss_input_dict.get("pointmap_scale", None)
            pointmap_shift = ss_input_dict.get("pointmap_shift", None)
            return_dict.update(
                self.pose_decoder(return_dict, scene_scale=pointmap_scale, scene_shift=pointmap_shift)
            )
            return_dict["scale"] = return_dict["scale"] * return_dict["downsample_factor"]
            
            ss_return_dict = return_dict
            
            # Clean up Stage 1 models
            delete_model_completely(ss_generator, "ss_generator")
            delete_model_completely(ss_decoder, "ss_decoder")
            delete_model_completely(ss_condition_embedder, "ss_condition_embedder")
            force_gc()
            log_memory("After Stage 1")
            
            # Early exit if stage1_only
            if stage1_only:
                _ck("stage1_sparse_structure")
                _ss_steps = stage1_inference_steps or self.ss_inference_steps
                _ss_mode = "shortcut-distilled" if use_stage1_distillation else "flow-matching+CFG"
                logger.info(f"[TIMING] Stage 1: {_ss_steps} steps, {_ss_mode}")
                logger.info(f"[TIMING] Peak RSS: {get_memory_gb():.1f} GB")
                
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return {
                    **ss_return_dict,
                    "pointmap": pts.cpu().permute((1, 2, 0)),
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),
                    "stage_timings": dict(_stage_secs),
                    "peak_rss_gb": get_memory_gb(),
                }
            
            # ────────────────────────────────────────────────────────────────────
            # STAGE 2: STRUCTURED LATENT (with optional MPS acceleration)
            # ────────────────────────────────────────────────────────────────────
            
            _ck("stage1_sparse_structure")
            logger.info("[S2] === STAGE 2: Structured Latent Generation ===")
            
            coords = ss_return_dict["coords"]
            stage2_device = get_stage2_device(use_stage2_mps)
            
            if stage2_device.type == "mps":
                logger.info("[S2] Stage 2 running on MPS (Metal GPU)")
            else:
                logger.info("[S2] Stage 2 running on CPU")
            
            # Load SLAT generator
            slat_generator, slat_condition_embedder = self._load_generator(
                "slat_generator_config_path", "slat_generator_ckpt_path"
            )
            
            # Move to Stage 2 device (MPS or CPU)
            slat_generator = slat_generator.to(stage2_device)
            if slat_condition_embedder is not None:
                slat_condition_embedder = slat_condition_embedder.to(stage2_device)
            
            # Configure Stage 2 generator
            slat_generator.inference_steps = stage2_inference_steps or self.slat_inference_steps
            slat_generator.reverse_fn.interval = self.slat_cfg_interval
            slat_generator.rescale_t = self.slat_rescale_t
            
            if use_stage2_distillation:
                slat_generator.no_shortcut = False
                slat_generator.reverse_fn.strength = 0
            else:
                slat_generator.no_shortcut = True
                slat_generator.reverse_fn.strength = self.slat_cfg_strength
            
            # Run Stage 2
            with torch.no_grad():
                with torch.inference_mode():
                    latent_shape = (slat_input_dict["image"].shape[0],) + (coords.shape[0], 8)
                    
                    # Move inputs and coords to stage2_device
                    for key in slat_input_dict:
                        if isinstance(slat_input_dict[key], torch.Tensor):
                            slat_input_dict[key] = slat_input_dict[key].to(stage2_device)
                    
                    coords_stage2 = coords.to(stage2_device) if isinstance(coords, torch.Tensor) else coords
                    
                    # Get condition embeddings
                    if slat_condition_embedder is not None:
                        cond_tokens = slat_condition_embedder(**slat_input_dict)
                        condition_args = (cond_tokens, coords_stage2.cpu().numpy())
                        condition_kwargs = {}
                    else:
                        condition_args = (coords_stage2.cpu().numpy(),)
                        condition_kwargs = slat_input_dict
                    
                    # Generate latent
                    slat_raw = slat_generator(
                        latent_shape,
                        stage2_device,
                        *condition_args,
                        **condition_kwargs,
                    )
                    
                    # Move SLAT back to base device
                    slat = sp.SparseTensor(coords=coords, feats=slat_raw[0]).to(self.device)
                    slat = slat * self.slat_std.to(self.device) + self.slat_mean.to(self.device)
            
            # Clean up Stage 2 models
            delete_model_completely(slat_generator, "slat_generator")
            delete_model_completely(slat_condition_embedder, "slat_condition_embedder")
            
            # Clean up MPS if it was used
            if stage2_device.type == "mps":
                force_gc()
            
            log_memory("After Stage 2")
            
            # Save SLAT to cache (single save location)
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                
                # Save with hash if available
                if input_hash:
                    slat_cache_path = get_cache_path(self.cache_dir, "stage2_slat", input_hash)
                else:
                    # Fallback: save with timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    slat_cache_path = os.path.join(self.cache_dir, f"slat_{timestamp}.pt")
                
                try:
                    save_cache({
                        "coords": slat.coords.cpu(),
                        "feats": slat.feats.cpu(),
                        "ss_return_dict_translation": ss_return_dict.get("translation"),
                        "ss_return_dict_scale": ss_return_dict.get("scale"),
                        "ss_return_dict_rotation": ss_return_dict.get("rotation"),
                        "intrinsics": intrinsics.cpu() if isinstance(intrinsics, torch.Tensor) else intrinsics,
                        "pts": pts.cpu() if pts is not None else None,
                        "pts_colors": pts_colors.cpu() if pts_colors is not None else None,
                    }, slat_cache_path)
                except Exception as e:
                    logger.warning(f"[CACHE] Failed to save SLAT: {e}")
        
        # ────────────────────────────────────────────────────────────────────────
        # STAGE 3: DECODING (mesh and gaussian)
        # ────────────────────────────────────────────────────────────────────────
        
        _ck("stage2_slat")
        logger.info("[S3] === STAGE 3: Decoding ===")
        
        formats = list(decode_formats or self.decode_formats)
        _need_gaussian = (
            (texture_bake and texture_bake_source == "gaussian")
            or (not texture_bake and vertex_color_source == "gaussian")
        )
        if "mesh" in formats and _need_gaussian and "gaussian" not in formats:
            formats.append("gaussian")
        
        outputs = {}
        
        # Mesh decoding
        if "mesh" in formats:
            logger.info("[S3] Decoding mesh...")
            slat_decoder_mesh = self._load_model(
                "slat_decoder_mesh_config_path", "slat_decoder_mesh_ckpt_path"
            ).float()
            
            slat_mesh = slat.float()
            slat_decoder_mesh = slat_decoder_mesh.to(slat_mesh.device)
            
            with torch.no_grad():
                meshes = slat_decoder_mesh(slat_mesh)
                outputs["mesh"] = meshes
                
                if len(meshes) > 0:
                    logger.info(
                        f"[S3] Decoded mesh: {meshes[0].vertices.shape[0]} vertices, "
                        f"{meshes[0].faces.shape[0]} faces"
                    )
            
            delete_model_completely(slat_decoder_mesh, "slat_decoder_mesh")
            force_gc()
            log_memory("After mesh decoding")
        
        # Gaussian decoding
        if "gaussian" in formats:
            logger.info("[S3] Decoding gaussians...")
            slat_decoder_gs = self._load_model(
                "slat_decoder_gs_config_path", "slat_decoder_gs_ckpt_path"
            ).float()
            
            if hasattr(slat_decoder_gs, "convert_to_fp32"):
                slat_decoder_gs.convert_to_fp32()
            
            slat_decoder_gs = slat_decoder_gs.to(slat.device)
            slat_gs = slat if slat.feats.dtype == torch.float32 else sp.SparseTensor(
                coords=slat.coords, feats=slat.feats.float()
            )
            
            with torch.no_grad():
                outputs["gaussian"] = slat_decoder_gs(slat_gs)
                if len(outputs["gaussian"]) > 0:
                    logger.info(f"[S3] Decoded {outputs['gaussian'][0].get_xyz.shape[0]} gaussians")
            
            delete_model_completely(slat_decoder_gs, "slat_decoder_gs")
            force_gc()
            log_memory("After gaussian decoding")
        
        # Post-processing
        if "mesh" in outputs:
            gaussian_rep = outputs.get("gaussian", [None])[0]
            
            # Determine color path
            do_bake = texture_bake
            bake_source = texture_bake_source
            if do_bake and bake_source == "gaussian" and gaussian_rep is None:
                logger.warning("[S3] Gaussian unavailable; falling back to vertex-color bake")
                bake_source = "vertex"
            
            if do_bake:
                app_rep = gaussian_rep if bake_source == "gaussian" else None
                vcolor_desc = None
            else:
                app_rep = gaussian_rep if vertex_color_source == "gaussian" else None
                vcolor_desc = (
                    "gaussian" if app_rep is not None
                    else ("mesh" if vertex_color_source == "gaussian" else vertex_color_source)
                )
            
            logger.info(
                f"[S3] Post-processing: simplify={simplify_ratio}, "
                f"texture_bake={do_bake}, bake_source={bake_source if do_bake else 'vertex'}"
            )
            _ck("mesh_decode")
            
            glb = postprocessing_utils.to_glb(
                app_rep,
                outputs["mesh"][0],
                simplify=simplify_ratio,
                texture_size=texture_size,
                verbose=True,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=do_bake,
                use_vertex_color=not do_bake,
                bake_backend="portable",
                bake_source=bake_source,
            )
            outputs["glb"] = glb
            _ck("export_bake")
        
        if "gaussian" in outputs:
            outputs["gs"] = outputs["gaussian"][0]
        
        log_memory("Pipeline end")
        
        # Timing summary
        _ck("finalize")
        _ss_steps = stage1_inference_steps or self.ss_inference_steps
        _slat_steps = stage2_inference_steps or self.slat_inference_steps
        _ss_mode = "shortcut-distilled" if use_stage1_distillation else "flow-matching+CFG"
        _slat_mode = "shortcut-distilled" if use_stage2_distillation else "flow-matching+CFG"
        _ss_nfe = _ss_steps if use_stage1_distillation else round(_ss_steps * 1.5)
        _slat_nfe = _slat_steps if use_stage2_distillation else round(_slat_steps * 1.5)
        
        _order = ["setup", "depth", "stage1_sparse_structure", "stage2_slat", "mesh_decode", "export_bake", "finalize"]
        _labels = {
            "setup": "Setup / preprocess",
            "depth": "S0 MoGe depth",
            "stage1_sparse_structure": f"S1 sparse structure ({_ss_steps} steps, {_ss_mode}, ~{_ss_nfe} NFE)",
            "stage2_slat": f"S2 SLAT texture ({_slat_steps} steps, {_slat_mode}, ~{_slat_nfe} NFE, {'MPS' if use_stage2_mps else 'CPU'})",
            "mesh_decode": "S3 mesh decode",
            "export_bake": "Export / bake",
            "finalize": "Finalize",
        }
        
        _total = sum(_stage_secs.values())
        logger.info("[TIMING] ==================== Per-stage wall-clock ====================")
        for _k in _order:
            if _k in _stage_secs:
                logger.info(f"[TIMING] {_labels[_k]:<50} {_stage_secs[_k]:>8.2f}s")
        logger.info(f"[TIMING] {'TOTAL':<50} {_total:>8.2f}s")
        logger.info(f"[TIMING] Peak RSS: {get_memory_gb():.1f} GB")
        logger.info("[TIMING] ==============================================================")
        
        # Assemble result
        result = {**ss_return_dict, **outputs}
        result["stage_timings"] = dict(_stage_secs)
        result["stage_timings"]["total"] = _total
        result["peak_rss_gb"] = get_memory_gb()
        
        if intrinsics is not None:
            result["intrinsics"] = intrinsics
        
        if pts is not None:
            result["pointmap"] = pts.cpu().permute((1, 2, 0))
            result["pointmap_colors"] = pts_colors.cpu().permute((1, 2, 0))
        
        return result
    
    def run_multi_view(
        self,
        images: list,
        masks: Optional[list] = None,
        seed: int = 42,
        stage1_only: bool = False,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_vertex_color: bool = False,
        stage1_inference_steps: Optional[int] = None,
        stage2_inference_steps: Optional[int] = None,
        use_stage1_distillation: bool = False,
        use_stage2_distillation: bool = False,
        use_stage2_mps: bool = True,
        decode_formats: Optional[list] = None,
        fusion_config: Optional[dict] = None,
    ) -> dict:
        """
        Multi-view 3D reconstruction with improved texture mapping and occlusion handling.
        
        Process:
        1. Reconstruct each view independently (full pipeline per view)
        2. Average sparse geometry coordinates (coordinates)
        3. Fuse appearance (gaussians) with per-view confidence weights
        4. Blend vertex colors from all views using visibility weighting
        5. Export final mesh with fused appearance
        
        Args:
            images: List of input images
            masks: List of corresponding masks (optional)
            fusion_config: Dict with keys:
                - view_weighting: 'uniform' (default) or 'entropy' (entropy-based weights)
        
        Returns:
            Fused 3D reconstruction result
        """
        if masks is None:
            masks = [None] * len(images)
        
        if len(images) < 2:
            logger.warning(f"[MV] Multi-view needs ≥2 images; using single-view (got {len(images)})")
            return self.run(
                images[0], masks[0], seed=seed, stage1_only=stage1_only,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=with_texture_baking,
                use_vertex_color=use_vertex_color,
                stage1_inference_steps=stage1_inference_steps,
                stage2_inference_steps=stage2_inference_steps,
                use_stage1_distillation=use_stage1_distillation,
                use_stage2_distillation=use_stage2_distillation,
                use_stage2_mps=use_stage2_mps,
                decode_formats=decode_formats,
            )
        
        logger.info(f"[MV] Multi-view reconstruction: {len(images)} views (improved fusion)")
        logger.info("[MV] Processing each view independently...")
        
        results = []
        for i, (img, m) in enumerate(zip(images, masks)):
            logger.info(f"[MV] View {i+1}/{len(images)}...")
            result = self.run(
                img, m, seed=seed + i, stage1_only=stage1_only,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=with_texture_baking,
                use_vertex_color=use_vertex_color,
                stage1_inference_steps=stage1_inference_steps,
                stage2_inference_steps=stage2_inference_steps,
                use_stage1_distillation=use_stage1_distillation,
                use_stage2_distillation=use_stage2_distillation,
                use_stage2_mps=use_stage2_mps,
                decode_formats=decode_formats,
            )
            results.append(result)
        
        # ─────────────────────────────────────────────────────────────────────────────
        # GEOMETRY FUSION (Coordinate averaging)
        # ─────────────────────────────────────────────────────────────────────────────
        logger.info("[MV] Step 1: Fusing sparse geometry coordinates...")
        coords_list = [r.get("coords") for r in results if "coords" in r]
        
        if coords_list:
            coords_tensors = [torch.as_tensor(c, dtype=torch.float32) for c in coords_list]
            fused_coords = torch.stack(coords_tensors).mean(dim=0).int()
            logger.info(f"[MV] ✓ Fused {len(coords_list)} coordinate sets: {fused_coords.shape}")
        else:
            fused_coords = results[0].get("coords")
            logger.warning("[MV] No coordinates to fuse; using first view")
        
        # ─────────────────────────────────────────────────────────────────────────────
        # APPEARANCE FUSION (Gaussians with confidence weighting)
        # ─────────────────────────────────────────────────────────────────────────────
        logger.info("[MV] Step 2: Computing per-view confidence weights...")
        confidence_weights = compute_view_confidence_weights(results, fusion_config)
        
        logger.info("[MV] Step 3: Fusing appearance from all views...")
        gaussian_list = [r.get("gs") or r.get("gaussian") for r in results]
        
        # Filter None values and fuse
        gaussian_list_valid = [g for g in gaussian_list if g is not None]
        if len(gaussian_list_valid) > 1:
            fused_gaussian = fuse_gaussians_with_confidence(gaussian_list_valid, confidence_weights)
            logger.info(f"[MV] ✓ Fused appearance from {len(gaussian_list_valid)} views")
        elif gaussian_list_valid:
            fused_gaussian = gaussian_list_valid[0]
            logger.info("[MV] Only one valid gaussian; using it as-is")
        else:
            fused_gaussian = None
            logger.warning("[MV] No gaussians to fuse; export will use base mesh only")
        
        # ─────────────────────────────────────────────────────────────────────────────
        # RESULT ASSEMBLY (Use base result with fused geometry and appearance)
        # ─────────────────────────────────────────────────────────────────────────────
        logger.info("[MV] Step 4: Assembling final result with fused data...")
        
        fused_result = results[0].copy()
        fused_result["coords"] = fused_coords
        
        # Update gaussian if fusion succeeded
        if fused_gaussian is not None:
            fused_result["gs"] = fused_gaussian
            if "gaussian" in fused_result:
                fused_result["gaussian"] = [fused_gaussian]
        
        # Helper to get values from dict or object
        def get_config_value(cfg, key, default):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            else:
                return getattr(cfg, key, default)
        
        # Add fusion metadata
        fused_result["multi_view_fusion"] = {
            "num_views": len(results),
            "confidence_weights": confidence_weights,
            "weighting_mode": get_config_value(fusion_config, "view_weighting", "uniform") if fusion_config else "uniform",
            "geometry_fused": True,
            "appearance_fused": fused_gaussian is not None,
        }
        
        logger.info("[MV] ✓ Multi-view fusion complete!")
        logger.info(f"[MV] Final result: coords {fused_coords.shape if isinstance(fused_coords, np.ndarray) else 'array'}, "
                   f"gaussians {'fused' if fused_gaussian else 'single-view'}")
        
        return fused_result
