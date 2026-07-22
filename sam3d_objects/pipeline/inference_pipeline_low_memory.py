# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Low-memory inference pipeline for SAM-3D.

This pipeline loads models sequentially and deletes them after use,
reducing peak memory from ~45GB to ~15GB.
"""

import os
import gc
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
    layout_post_optimization,
)
from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils


def force_gc():
    """Aggressive garbage collection and cache clearing."""
    gc.collect()
    gc.collect()
    gc.collect()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def delete_model_completely(model, name="model"):
    """Fully delete a model and all its parameters from memory."""
    if model is None:
        return
    
    try:
        # Move to CPU first
        model.cpu()
        
        # Delete all parameters
        for param in model.parameters():
            param.data = torch.empty(0)
            if param.grad is not None:
                param.grad = None
        
        # Delete all buffers
        for buffer_name, buffer in list(model.named_buffers()):
            buffer.data = torch.empty(0)
        
        # Clear any cached properties
        if hasattr(model, '_modules'):
            model._modules.clear()
        
        del model
        logger.info(f"[LOW-MEM] Deleted {name}")
    except Exception as e:
        logger.warning(f"[LOW-MEM] Failed to delete {name}: {e}")
    
    force_gc()


def get_memory_gb():
    """Get current memory usage in GB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 ** 3)
    except:
        return -1


def log_memory(stage: str):
    """Log current memory usage."""
    mem = get_memory_gb()
    if mem > 0:
        logger.info(f"[LOW-MEM] {stage}: {mem:.1f} GB")


# =============================================================================
# CACHING UTILITIES
# =============================================================================

def save_cache(data: dict, cache_path: str):
    """Save intermediate outputs to cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data, cache_path)
    logger.info(f"[CACHE] Saved to {cache_path}")

def load_cache(cache_path: str) -> Optional[dict]:
    """Load intermediate outputs from cache file."""
    if os.path.exists(cache_path):
        logger.info(f"[CACHE] Loading from {cache_path}")
        return torch.load(cache_path, weights_only=False)
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


class InferencePipelineLowMemory:
    """
    Low-memory version of InferencePipelinePointMap.
    
    Key difference: Models are loaded on-demand and deleted after each stage.
    This reduces peak memory from ~45GB to ~15GB at the cost of loading time.
    """
    
    def __init__(
        self,
        config_path: str,
        depth_model=None,
        layout_post_optimization_method=None,
        clip_pointmap_beyond_scale=None,
        device="cpu",
        dtype="float16",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the low-memory pipeline.
        
        Unlike the regular pipeline, models are NOT loaded here.
        They are loaded on-demand during run().
        
        Args:
            cache_dir: Optional directory for caching intermediate outputs.
                      If provided, stages can be skipped if cache exists.
        """
        self.config_path = config_path
        self.workspace_dir = os.path.dirname(config_path)
        self.config = OmegaConf.load(config_path)
        self.device = torch.device(device)
        self.dtype = self._get_dtype(dtype)
        # Layout post-optimization method (pose refinement against the pointmap +
        # mask). Defaults to the CPU/MPS-portable implementation so callers can
        # opt in via run(with_layout_postprocess=True) without extra wiring.
        self.layout_post_optimization_method = (
            layout_post_optimization_method or layout_post_optimization
        )
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        self.cache_dir = cache_dir
        
        # Pipeline settings from config
        self.decode_formats = self.config.get("decode_formats", ["mesh"])
        self.pad_size = self.config.get("pad_size", 1.0)
        self.version = self.config.get("version", "v0")
        self.downsample_ss_dist = self.config.get("downsample_ss_dist", 0)
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
        
        # Initialize preprocessors (lightweight, keep in memory)
        self.ss_preprocessor = instantiate(self.config.ss_preprocessor)
        self.slat_preprocessor = instantiate(self.config.slat_preprocessor)
        
        # Pose decoder (lightweight)
        pose_decoder_name = self.config.get("pose_decoder_name", "default")
        self.pose_decoder = get_pose_decoder(pose_decoder_name)
        
        # Store depth model config but don't instantiate yet
        self.depth_model_config = self.config.get("depth_model", None)
        self.depth_model = depth_model  # Can be pre-loaded externally
        
        # Track what's currently loaded
        self._loaded_models = {}
        
        if cache_dir:
            logger.info(f"[LOW-MEM] Pipeline initialized with cache at: {cache_dir}")
        else:
            logger.info(f"[LOW-MEM] Pipeline initialized (no caching)")
        log_memory("After init")
    
    @staticmethod
    def _get_dtype(dtype):
        if dtype == "bfloat16":
            return torch.bfloat16
        elif dtype == "float16":
            return torch.float16
        elif dtype == "float32":
            return torch.float32
        else:
            raise NotImplementedError(f"Unknown dtype: {dtype}")
    
    def _load_model(self, config_key: str, ckpt_key: str, state_dict_fn=None):
        """Load a model on-demand."""
        config_path = os.path.join(self.workspace_dir, self.config[config_key])
        ckpt_path = os.path.join(self.workspace_dir, self.config[ckpt_key])
        
        logger.info(f"[LOW-MEM] Loading {config_key}...")
        config = OmegaConf.load(config_path)
        
        # Remove pretrained path if present (we're loading separately)
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        
        model = instantiate(config)
        
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
        
        model = model.to(self.device)
        model.eval()
        
        log_memory(f"After loading {config_key}")
        return model
    
    def _load_generator(self, config_key: str, ckpt_key: str):
        """Load a generator model with condition embedder."""
        config_path = os.path.join(self.workspace_dir, self.config[config_key])
        ckpt_path = os.path.join(self.workspace_dir, self.config[ckpt_key])
        
        logger.info(f"[LOW-MEM] Loading {config_key} with embedder...")
        full_config = OmegaConf.load(config_path)
        
        # Load generator backbone
        gen_config = full_config["module"]["generator"]["backbone"]
        state_dict_fn = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")
        
        generator = instantiate(gen_config)
        
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device="cpu")
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
        
        gen_state_dict = state_dict_fn(state_dict.copy())
        generator.load_state_dict(gen_state_dict, strict=False)
        generator = generator.to(self.device)
        generator.eval()
        
        # Load condition embedder
        cond_embedder = None
        if "condition_embedder" in full_config["module"]:
            cond_config = full_config["module"]["condition_embedder"]["backbone"]
            cond_state_dict_fn = filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder.")
            cond_embedder = instantiate(cond_config)
            cond_state_dict = cond_state_dict_fn(state_dict.copy())
            cond_embedder.load_state_dict(cond_state_dict, strict=False)
            cond_embedder = cond_embedder.to(self.device)
            cond_embedder.eval()
        
        log_memory(f"After loading {config_key}")
        return generator, cond_embedder
    
    def _load_depth_model(self):
        """Load the depth model (MoGe) on demand."""
        if self.depth_model is not None:
            return self.depth_model
        
        if self.depth_model_config is None:
            raise ValueError("No depth model config provided")
        
        logger.info("[LOW-MEM] Loading depth model (MoGe)...")
        self.depth_model = instantiate(self.depth_model_config)
        
        # Move to MPS for GPU acceleration (instead of CPU)
        if torch.backends.mps.is_available() and hasattr(self.depth_model, 'model'):
            logger.info("[LOW-MEM] Moving depth model to MPS for GPU acceleration")
            self.depth_model.device = torch.device("mps")
            self.depth_model.model.to("mps")
        
        log_memory("After loading depth model")
        return self.depth_model
    
    def _unload_depth_model(self):
        """Unload the depth model."""
        if self.depth_model is not None:
            if hasattr(self.depth_model, 'model'):
                delete_model_completely(self.depth_model.model, "depth_model.model")
            self.depth_model = None
            force_gc()
            log_memory("After unloading depth model")
    
    def image_to_float(self, image):
        image = np.array(image)
        image = image / 255
        image = image.astype(np.float32)
        return image
    
    def merge_image_and_mask(self, image, mask):
        """Merge image and mask into RGBA format.
        
        Properly converts boolean masks to uint8 format (0-255).
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = np.array(image)
        
        if mask is not None:
            mask = np.array(mask)
            # Convert boolean mask to uint8 (0 or 255)
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            # Ensure mask is uint8 with proper range
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            if mask.ndim == 2:
                mask = mask[..., None]
            # Combine RGB with mask as alpha
            image = np.concatenate([image[..., :3], mask], axis=-1)
        
        return image

    
    def compute_pointmap(self, image):
        """Compute pointmap using MoGe depth model."""
        loaded_image = self.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_mask = loaded_image[..., -1]
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]
        
        depth_model = self._load_depth_model()
        
        with torch.no_grad():
            with torch.inference_mode():
                output = depth_model(loaded_image)
        
        # Move pointmaps to CPU for pytorch3d Transform3d (MPS not supported)
        pointmaps = output["pointmaps"].float().cpu()
        camera_convention_transform = (
            Transform3d()
            .rotate(camera_to_pytorch3d_camera(device="cpu").rotation)
            .to("cpu")
        )
        points_tensor = camera_convention_transform.transform_points(pointmaps)
        intrinsics = output.get("intrinsics", None)
        if intrinsics is not None:
            intrinsics = intrinsics.cpu() if hasattr(intrinsics, 'cpu') else intrinsics
        
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
    
    def preprocess_image(self, image, preprocessor, pointmap=None):
        """Preprocess image for model input."""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")
        
        preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap
        )
        
        _item = preprocessor_return_dict
        item = {
            "mask": _item["mask"][None].to(self.device),
            "image": _item["image"][None].to(self.device),
            "rgb_image": _item["rgb_image"][None].to(self.device),
            "rgb_image_mask": _item["rgb_image_mask"][None].to(self.device),
        }
        
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = _item["pointmap"][None].to(self.device)
            item["rgb_pointmap"] = _item["rgb_pointmap"][None].to(self.device)
            item["pointmap_scale"] = _item["pointmap_scale"][None].to(self.device)
            item["pointmap_shift"] = _item["pointmap_shift"][None].to(self.device)
            item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"][None].to(self.device)
            item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"][None].to(self.device)
        
        return item
    
    @staticmethod
    def _down_sample_img(img_3chw: torch.Tensor):
        x = img_3chw.unsqueeze(0)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        max_side = max(x.shape[2], x.shape[3])
        scale_factor = 1.0
        if max_side > 3800:
            scale_factor = 0.125
        elif max_side > 1900:
            scale_factor = 0.25
        elif max_side > 1200:
            scale_factor = 0.5
        x = torch.nn.functional.interpolate(
            x, scale_factor=(scale_factor, scale_factor),
            mode="bilinear", align_corners=False, antialias=True,
        )
        return x.squeeze(0)
    
    def _pose_to_placement_transform(self, pose, device):
        """Build the local->camera placement transform from a decoded pose.

        Mirrors the convention used by the layout optimizer (get_mesh): the mesh
        is first rotated from z-up (mesh decoder frame) to y-up (pytorch3d camera
        frame), then the predicted scale/rotation/translation are applied.

        Returns a pytorch3d Transform3d (batch of 1).
        """
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

        quat = pose["rotation"].to(device).float()
        trans = pose["translation"].to(device).float()
        scale = pose["scale"].to(device).float()
        # Normalise shapes to a batch of 1.
        if quat.dim() == 1:
            quat = quat[None]
        quat = quat.reshape(-1, 4)[:1]
        trans = trans.reshape(-1, 3)[:1]
        scale = scale.reshape(-1, 3)[:1]

        rotation = quaternion_to_matrix(quat)  # (1, 3, 3)

        # z-up -> y-up basis change, applied as verts @ R_convert.T (row vectors).
        r_convert = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        convert_tfm = Transform3d(device=device).rotate(r_convert.T[None])
        pose_tfm = compose_transform(
            scale=scale, rotation=rotation, translation=trans
        )
        return convert_tfm.compose(pose_tfm)

    def _apply_transform_to_glb(self, glb, transform, device):
        """Apply a pytorch3d Transform3d to every geometry in a trimesh GLB.

        trimesh uses column-vector homogeneous matrices (v' = M @ v), whereas
        pytorch3d's get_matrix() is row-vector (v' = v @ M); the two differ by a
        transpose. Returns a transformed copy (input is left untouched).
        """
        m_row = transform.get_matrix()[0].detach().cpu().numpy()  # (4, 4) row-vector
        m_col = m_row.T  # column-vector convention for trimesh
        placed = deepcopy(glb)
        placed.apply_transform(m_col)
        return placed

    def _run_layout_placement(
        self,
        glb,
        pose,
        intrinsics,
        layout_mask,
        layout_pointmap,
        refine,
        device,
    ):
        """Refine the object pose against the pointmap/mask and place the mesh.

        Args:
            glb: canonical-space trimesh produced by ``to_glb``.
            pose: dict with ``rotation`` (quat), ``translation``, ``scale``.
            intrinsics: normalised 3x3 camera intrinsics tensor (or None).
            layout_mask: HxW object mask at model-input resolution (or None).
            layout_pointmap: HxWx3 pointmap at model-input resolution (or None).
            refine: if True and inputs are available, run the ICP + render-compare
                layout optimizer to refine the pose before placing.
            device: torch device string for the (CPU/MPS) optimizer.

        Returns:
            (placed_glb, refined_pose, iou). ``placed_glb`` is a copy of ``glb``
            transformed into camera space; ``refined_pose`` is the (possibly
            refined) pose dict; ``iou`` is the layout IoU (or None).
        """
        refined_pose = dict(pose)
        iou = None

        can_refine = (
            refine
            and self.layout_post_optimization_method is not None
            and intrinsics is not None
            and layout_mask is not None
            and layout_pointmap is not None
        )
        if can_refine:
            try:
                logger.info("[LOW-MEM] Running layout post-optimization (pose refine)...")
                intr = intrinsics.clone().float().to(device)
                fx, fy = intr[0, 0], intr[1, 1]
                re_focal = torch.minimum(fx, fy)
                intr[0, 0], intr[1, 1] = re_focal, re_focal
                (
                    revised_quat,
                    revised_t,
                    revised_scale,
                    final_iou,
                    _flag_icp,
                    _flag_optim,
                ) = self.layout_post_optimization_method(
                    deepcopy(glb),
                    pose["rotation"].to(device),
                    pose["translation"].to(device),
                    pose["scale"].to(device),
                    layout_mask.to(device),
                    layout_pointmap.to(device),
                    intr,
                    min_size=518,
                    device=torch.device(device),
                )
                refined_pose = {
                    "rotation": revised_quat,
                    "translation": revised_t,
                    "scale": revised_scale,
                }
                iou = final_iou
                logger.info(f"[LOW-MEM] Layout refinement IoU: {final_iou}")
            except Exception as e:  # never break the main path on a refine failure
                logger.warning(
                    f"[LOW-MEM] Layout refinement failed ({e}); placing with decoded pose",
                    exc_info=True,
                )
                refined_pose = dict(pose)

        try:
            transform = self._pose_to_placement_transform(refined_pose, device)
            placed_glb = self._apply_transform_to_glb(glb, transform, device)
        except Exception as e:
            logger.warning(f"[LOW-MEM] Object placement failed ({e}); returning canonical mesh")
            placed_glb = None

        return placed_glb, refined_pose, iou

    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=False,
        layout_refine=False,
        use_vertex_color=True,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        decode_formats=None,
        use_cache: bool = True,
        simplify_ratio: float = 0.0,
        load_slat: str = None,
        texture_bake: bool = False,
        texture_bake_source: str = "gaussian",
        texture_size: int = 2048,
        vertex_color_source: str = "gaussian",
    ) -> dict:
        """
        Run the full inference pipeline with sequential model loading.
        
        Each model is loaded, used, then deleted before loading the next.
        
        Args:
            use_cache: If True and cache_dir is set, load cached stages and save new results.
            simplify_ratio: Ratio of triangles to remove during mesh simplification (0.0=none, 0.95=heavy).
            load_slat: Path to a cached SLAT .pt file to load directly (skips stages 0-2).
            with_layout_postprocess: If True, place the decoded object into camera
                space using the predicted pose. Emits ``glb_placed`` in the output.
            layout_refine: If True (and layout inputs are available), refine the pose
                with the ICP + render-compare layout optimizer before placement.
                Requires a fresh run (not a cached SLAT). Slower; runs on CPU.
            use_stage1_distillation / use_stage2_distillation: If True, sample the
                corresponding flow stage with the shortcut model (step-size
                conditioning, CFG off, ~1 network eval per step) instead of
                CFG-guided flow matching. Faster with far fewer steps, but requires
                shortcut-distilled weights (is_shortcut_model=True in the generator
                config). See arXiv:2410.12557.
        """
        logger.info("[LOW-MEM] Starting sequential inference pipeline")
        log_memory("Start of run()")
        
        image = self.merge_image_and_mask(image, mask)
        
        # Compute input hash for caching
        input_hash = None
        slat_from_cache = False
        slat = None
        ss_return_dict = {}
        pts = None
        pts_colors = None
        # Layout-placement state (persisted across stages). intrinsics comes from
        # STAGE 0; the mask/pointmap for optional pose refinement come from the SS
        # preprocessor and are only available on a fresh (non-cached) run.
        intrinsics = None
        layout_mask = None
        layout_pointmap = None
        
        # Check if we should load SLAT from a specific file
        if load_slat:
            logger.info(f"[CACHE] Loading SLAT from: {load_slat}")
            slat_cache = torch.load(load_slat, map_location="cpu")
            slat = sp.SparseTensor(
                coords=slat_cache["coords"].to(self.device),
                feats=slat_cache["feats"].to(self.device)
            )
            ss_return_dict["translation"] = slat_cache.get("ss_return_dict_translation")
            ss_return_dict["scale"] = slat_cache.get("ss_return_dict_scale")
            ss_return_dict["coords"] = slat_cache.get("coords")
            ss_return_dict["rotation"] = slat_cache.get("ss_return_dict_rotation")
            intrinsics = slat_cache.get("intrinsics")
            pts = slat_cache.get("pts")
            pts_colors = slat_cache.get("pts_colors")
            slat_from_cache = True
            logger.info(f"[CACHE] Loaded SLAT with {slat.coords.shape[0]} voxels - skipping Stages 0, 1, 2!")
        elif self.cache_dir and use_cache:
            np_image = np.array(image) if isinstance(image, Image.Image) else image
            np_mask = np.array(mask) if isinstance(mask, (Image.Image, np.ndarray)) else np.zeros((1,))
            input_hash = compute_input_hash(np_image, np_mask)
            logger.info(f"[CACHE] Input hash: {input_hash}")
            
            # Check if SLAT is cached (allows skipping Stages 0, 1, 2)
            slat_cache_path = get_cache_path(self.cache_dir, "stage2_slat", input_hash)
            slat_cache = load_cache(slat_cache_path)
            if slat_cache is not None:
                logger.info("[CACHE] Found cached SLAT - skipping Stages 0, 1, 2!")
                slat = sp.SparseTensor(
                    coords=slat_cache["coords"].to(self.device),
                    feats=slat_cache["feats"].to(self.device)
                )
                ss_return_dict["translation"] = slat_cache.get("ss_return_dict_translation")
                ss_return_dict["scale"] = slat_cache.get("ss_return_dict_scale")
                ss_return_dict["coords"] = slat_cache.get("coords")
                ss_return_dict["rotation"] = slat_cache.get("ss_return_dict_rotation")
                intrinsics = slat_cache.get("intrinsics")
                pts = slat_cache.get("pts")
                pts_colors = slat_cache.get("pts_colors")
                slat_from_cache = True
        
        # Skip Stages 0, 1, 2 if SLAT was loaded from cache
        if not slat_from_cache:
            # ========================
            # STAGE 0: Depth estimation
            # ========================
            logger.info("[LOW-MEM] === STAGE 0: Depth Estimation ===")
            
            pointmap_dict = self.compute_pointmap(image)
            pointmap = pointmap_dict["pointmap"]
            pts = self._down_sample_img(pointmap)
            pts_colors = self._down_sample_img(pointmap_dict["pts_color"])
            # Camera intrinsics for optional layout placement / pose refinement.
            intrinsics = pointmap_dict.get("intrinsics")
            
            # Unload depth model immediately
            self._unload_depth_model()
            log_memory("After depth stage complete")
            
            # Preprocess images
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor, pointmap=pointmap)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            
            # Stash the model-resolution mask + pointmap for optional pose
            # refinement (only available on a fresh run, not from SLAT cache).
            if with_layout_postprocess and layout_refine:
                _rgb_mask = ss_input_dict.get("rgb_image_mask")
                _rgb_pm = ss_input_dict.get("rgb_pointmap")
                if _rgb_mask is not None and _rgb_pm is not None:
                    layout_mask = _rgb_mask[0, 0].detach().cpu()
                    layout_pointmap = _rgb_pm[0].permute(1, 2, 0).detach().cpu()
            
            if seed is not None:
                torch.manual_seed(seed)
            
            # ========================
            # STAGE 1: Sparse Structure
            # ========================
            logger.info("[LOW-MEM] === STAGE 1: Sparse Structure Generation ===")
            
            # Load SS generator and embedder
            ss_generator, ss_condition_embedder = self._load_generator(
                "ss_generator_config_path", "ss_generator_ckpt_path"
            )
            
            # Load SS decoder
            ss_decoder = self._load_model("ss_decoder_config_path", "ss_decoder_ckpt_path")
            
            # Configure generator
            inference_steps = stage1_inference_steps or self.ss_inference_steps
            ss_generator.inference_steps = inference_steps
            ss_generator.reverse_fn.interval = self.ss_cfg_interval
            ss_generator.rescale_t = self.ss_rescale_t
            ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
            ss_generator.reverse_fn.unconditional_handling = "add_flag"
            # Shortcut-model distillation (arXiv:2410.12557): the shortcut sampler
            # conditions on step size d = 1/steps and runs CFG-free (1 NFE/step), so
            # it needs far fewer steps. Default is CFG-guided flow matching, which is
            # higher fidelity but 2 NFEs/step. Mirrors upstream's use_distillation.
            if use_stage1_distillation:
                ss_generator.no_shortcut = False
                ss_generator.reverse_fn.strength = 0
                ss_generator.reverse_fn.strength_pm = 0
            else:
                ss_generator.no_shortcut = True
                ss_generator.reverse_fn.strength = self.ss_cfg_strength
                ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm
            
            # Run stage 1
            with torch.no_grad():
                with torch.inference_mode():
                    bs = ss_input_dict["image"].shape[0]
                    
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
                    
                    return_dict = ss_generator(
                        latent_shape_dict,
                        ss_input_dict["image"].device,
                        *condition_args,
                        **condition_kwargs,
                    )
                    
                    if not hasattr(ss_generator.reverse_fn.backbone, "latent_mapping"):
                        return_dict = {"shape": return_dict}
                    
                    shape_latent = return_dict["shape"]
                    ss = ss_decoder(
                        shape_latent.permute(0, 2, 1).contiguous()
                        .view(shape_latent.shape[0], 8, 16, 16, 16)
                    )
                    coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
                    
                    return_dict["coords_original"] = coords
                    original_shape = coords.shape
                    if self.downsample_ss_dist > 0:
                        coords = prune_sparse_structure(coords, max_neighbor_axes_dist=self.downsample_ss_dist)
                    coords, downsample_factor = downsample_sparse_structure(coords)
                    logger.info(f"Downsampled coords from {original_shape[0]} to {coords.shape[0]}")
                    return_dict["coords"] = coords
                    return_dict["downsample_factor"] = downsample_factor
            
            # Run pose decoder
            pointmap_scale = ss_input_dict.get("pointmap_scale", None)
            pointmap_shift = ss_input_dict.get("pointmap_shift", None)
            return_dict.update(self.pose_decoder(return_dict, scene_scale=pointmap_scale, scene_shift=pointmap_shift))
            return_dict["scale"] = return_dict["scale"] * return_dict["downsample_factor"]
            
            ss_return_dict = return_dict
            
            # Unload stage 1 models
            delete_model_completely(ss_generator, "ss_generator")
            delete_model_completely(ss_decoder, "ss_decoder")
            delete_model_completely(ss_condition_embedder, "ss_condition_embedder")
            ss_generator = ss_decoder = ss_condition_embedder = None
            force_gc()
            log_memory("After Stage 1 complete")
            
            if stage1_only:
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return {
                    **ss_return_dict,
                    "pointmap": pts.cpu().permute((1, 2, 0)),
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),
                }
            
            # ========================
            # STAGE 2: Structured Latent
            # ========================
            logger.info("[LOW-MEM] === STAGE 2: Structured Latent Generation ===")
            
            coords = ss_return_dict["coords"]
            
            # Load SLAT generator
            slat_generator, slat_condition_embedder = self._load_generator(
                "slat_generator_config_path", "slat_generator_ckpt_path"
            )
            
            # Move SLAT generator to MPS for GPU acceleration (critical for performance)
            if torch.backends.mps.is_available():
                mps_device = torch.device("mps")
                logger.info(f"[LOW-MEM] Moving SLAT generator to MPS for GPU acceleration")
                slat_generator.to(mps_device)
                if slat_condition_embedder is not None:
                    slat_condition_embedder.to(mps_device)
            
            # Configure generator
            inference_steps = stage2_inference_steps or self.slat_inference_steps
            slat_generator.inference_steps = inference_steps
            slat_generator.reverse_fn.interval = self.slat_cfg_interval
            slat_generator.rescale_t = self.slat_rescale_t
            # See stage 1: shortcut distillation vs CFG-guided flow matching.
            if use_stage2_distillation:
                slat_generator.no_shortcut = False
                slat_generator.reverse_fn.strength = 0
            else:
                slat_generator.no_shortcut = True
                slat_generator.reverse_fn.strength = self.slat_cfg_strength
            
            # Run stage 2
            with torch.no_grad():
                with torch.inference_mode():
                    latent_shape = (slat_input_dict["image"].shape[0],) + (coords.shape[0], 8)
                    
                    # Config has slat_condition_input_mapping: [] (empty list)
                    # This means all inputs go as kwargs to the embedder
                    if slat_condition_embedder is not None:
                        # Move all tensor inputs to MPS to match embedder device
                        if torch.backends.mps.is_available():
                            mps_device = torch.device("mps")
                            for key, value in slat_input_dict.items():
                                if isinstance(value, torch.Tensor):
                                    slat_input_dict[key] = value.to(mps_device)
                        # Pass all inputs as kwargs (matching config with empty mapping)
                        cond_tokens = slat_condition_embedder(**slat_input_dict)
                        condition_args = (cond_tokens, coords.cpu().numpy())
                        condition_kwargs = {}
                    else:
                        # When no embedder, just pass coords
                        condition_args = (coords.cpu().numpy(),)
                        condition_kwargs = slat_input_dict
                    
                    slat_raw = slat_generator(
                        latent_shape,
                        slat_input_dict["image"].device,
                        *condition_args,
                        **condition_kwargs,
                    )
                    
                    slat = sp.SparseTensor(coords=coords, feats=slat_raw[0]).to(self.device)
                    slat = slat * self.slat_std.to(self.device) + self.slat_mean.to(self.device)
            
            # Unload stage 2 models
            delete_model_completely(slat_generator, "slat_generator")
            delete_model_completely(slat_condition_embedder, "slat_condition_embedder")
            slat_generator = slat_condition_embedder = None
            force_gc()
            log_memory("After Stage 2 complete")
            
            # Always save SLAT with timestamp for easy experimentation
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            slat_save_name = f"slat_{timestamp}.pt"
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                slat_save_path = os.path.join(self.cache_dir, slat_save_name)
            else:
                slat_save_path = slat_save_name
                
            torch.save({
                "coords": slat.coords.cpu(),
                "feats": slat.feats.cpu(),
                "ss_return_dict_translation": ss_return_dict.get("translation"),
                "ss_return_dict_scale": ss_return_dict.get("scale"),
                "ss_return_dict_rotation": ss_return_dict.get("rotation"),
                "intrinsics": intrinsics.cpu() if isinstance(intrinsics, torch.Tensor) else intrinsics,
                "pts": pts.cpu() if pts is not None else None,
                "pts_colors": pts_colors.cpu() if pts_colors is not None else None,
            }, slat_save_path)
            logger.info(f"[CACHE] Saved SLAT to: {slat_save_path}")
            
            # Also save to cache_dir if configured
            if self.cache_dir and use_cache and input_hash:
                slat_cache_path = get_cache_path(self.cache_dir, "stage2_slat", input_hash)
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
        
        # ========================
        # STAGE 3: Decoding
        # ========================
        logger.info("[LOW-MEM] === STAGE 3: Decoding ===")
        
        formats = list(decode_formats or self.decode_formats)
        # A Gaussian-sourced texture bake needs the Gaussian appearance rep decoded
        # alongside the mesh. Per-vertex color from the Gaussian needs it too.
        _need_gaussian = (
            (texture_bake and texture_bake_source == "gaussian")
            or (not texture_bake and vertex_color_source == "gaussian")
        )
        if "mesh" in formats and _need_gaussian and "gaussian" not in formats:
            formats.append("gaussian")
        outputs = {}
        
        if "mesh" in formats:
            # Chunked mesh decoding to fit in 48GB RAM
            # The mesh decoder expands 25K→1.5M voxels, requiring ~60GB
            # By chunking spatially, we reduce peak memory to ~20GB per chunk
            
            logger.info("[LOW-MEM] Loading mesh decoder...")
            slat_decoder_mesh = self._load_model(
                "slat_decoder_mesh_config_path", "slat_decoder_mesh_ckpt_path"
            )
            # Ensure float32 for maximum quality and MPS compatibility
            slat_decoder_mesh = slat_decoder_mesh.float()
            
            # Move mesh decoder to MPS for GPU acceleration
            # Added synchronization in upsample blocks to prevent OOM
            if torch.backends.mps.is_available():
                mps_device = torch.device("mps")
                logger.info("[LOW-MEM] Moving mesh decoder to MPS for GPU acceleration")
                slat_decoder_mesh.to(mps_device)
                # Ensure SLAT is float32 for maximum decoder quality
                slat = slat.to(device=mps_device, dtype=torch.float32)
            
            with torch.no_grad():
                # Single-pass decode in float32 for maximum quality
                # The memory-safe attention fix in masked_sdpa.py handles the MPS buffer limits
                meshes = slat_decoder_mesh(slat)
                outputs["mesh"] = meshes
                
                if len(meshes) > 0:
                    logger.info(f"[LOW-MEM] Decoded mesh: {meshes[0].vertices.shape[0]} vertices, {meshes[0].faces.shape[0]} faces")
                else:
                    logger.warning("[LOW-MEM] Mesh decoding failed to produce results!")
            
            delete_model_completely(slat_decoder_mesh, "slat_decoder_mesh")
            slat_decoder_mesh = None
            force_gc()
            log_memory("After mesh decoding")
        
        if "gaussian" in formats:
            # Load GS decoder. Unlike the mesh decoder this is light (a transformer
            # over the sparse SLAT tokens, no full-res voxel expansion).
            logger.info("[LOW-MEM] Loading gaussian decoder...")
            slat_decoder_gs = self._load_model(
                "slat_decoder_gs_config_path", "slat_decoder_gs_ckpt_path"
            ).float()
            # The base SparseTransformer forward casts activations to self.dtype
            # (float16 per config); .float() converts params but not that flag, so
            # also flip the torso to fp32 to avoid a Half/Float matmul mismatch
            # (which surfaces as a hard Metal assertion on MPS).
            if hasattr(slat_decoder_gs, "convert_to_fp32"):
                slat_decoder_gs.convert_to_fp32()

            # Decode on the same device the SLAT already lives on (MPS after the
            # mesh branch); the decoder's perturbation buffer is auto-registered on
            # that device too, so staying there avoids CPU/MPS tensor mixing.
            slat_decoder_gs.to(slat.device)
            slat_gs = slat if slat.feats.dtype == torch.float32 else sp.SparseTensor(
                coords=slat.coords, feats=slat.feats.float()
            )

            with torch.no_grad():
                outputs["gaussian"] = slat_decoder_gs(slat_gs)
                if len(outputs["gaussian"]) > 0:
                    logger.info(
                        f"[LOW-MEM] Decoded gaussian: {outputs['gaussian'][0].get_xyz.shape[0]} gaussians"
                    )

            delete_model_completely(slat_decoder_gs, "slat_decoder_gs")
            slat_decoder_gs = None
            force_gc()
            log_memory("After gaussian decoding")
        
        # Post-process outputs
        if "mesh" in outputs:
            gaussian_rep = outputs.get("gaussian", [None])[0]

            # Decide the color path (all portable — no CUDA rasterizer):
            #   texture_bake=False  -> per-vertex color on the mesh.
            #   texture_bake=True   -> bake a UV texture atlas, from the Gaussian
            #                          color field (source="gaussian") or from mesh
            #                          vertex colors (source="vertex").
            do_bake = texture_bake
            bake_source = texture_bake_source
            if do_bake and bake_source == "gaussian" and gaussian_rep is None:
                logger.warning(
                    "[LOW-MEM] Gaussian rep unavailable; falling back to vertex-color bake"
                )
                bake_source = "vertex"

            if do_bake:
                # Baking: app_rep is the Gaussian only for a gaussian-source bake.
                app_rep = gaussian_rep if bake_source == "gaussian" else None
                vcolor_desc = None
            else:
                # Per-vertex color: pass the Gaussian so to_glb colors vertices from
                # its (saturated) appearance field instead of the washed-out mesh
                # decoder head. Falls back to decoder colors if unavailable.
                app_rep = gaussian_rep if vertex_color_source == "gaussian" else None
                vcolor_desc = (
                    "gaussian" if app_rep is not None
                    else ("mesh" if vertex_color_source == "gaussian" else vertex_color_source)
                )

            logger.info(
                f"[LOW-MEM] Post-processing mesh (simplify={simplify_ratio}, "
                f"texture_bake={do_bake}, "
                f"source={bake_source if do_bake else 'vertex-color:' + str(vcolor_desc)})..."
            )
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
        
        if "gaussian" in outputs:
            outputs["gs"] = outputs["gaussian"][0]
        
        # ========================
        # Layout placement (opt-in)
        # ========================
        # Position the reconstructed object into camera space using the pose the
        # SS generator predicted (optionally refined against the pointmap/mask).
        # Emits ``glb_placed`` alongside the canonical ``glb``; the canonical mesh
        # is always kept so the default viewer/export path is unchanged.
        if with_layout_postprocess and outputs.get("glb") is not None:
            has_pose = all(
                ss_return_dict.get(k) is not None
                for k in ("rotation", "translation", "scale")
            )
            if not has_pose:
                logger.warning(
                    "[LOW-MEM] Layout placement requested but no decoded pose available; "
                    "skipping (the model config may not predict pose)."
                )
            else:
                pose = {
                    "rotation": ss_return_dict["rotation"],
                    "translation": ss_return_dict["translation"],
                    "scale": ss_return_dict["scale"],
                }
                placed_glb, refined_pose, layout_iou = self._run_layout_placement(
                    outputs["glb"],
                    pose,
                    intrinsics,
                    layout_mask,
                    layout_pointmap,
                    refine=layout_refine,
                    device="cpu",
                )
                if placed_glb is not None:
                    outputs["glb_placed"] = placed_glb
                    ss_return_dict.update(refined_pose)
                    if layout_iou is not None:
                        outputs["layout_iou"] = layout_iou
                    logger.info("[LOW-MEM] Object placed into camera space (glb_placed).")
        
        log_memory("End of run()")
        logger.info("[LOW-MEM] Pipeline complete!")
        
        result = {
            **ss_return_dict,
            **outputs,
        }
        if intrinsics is not None:
            result["intrinsics"] = intrinsics
        
        # Only include pointmap if it was computed (not when loading from cache)
        if pts is not None:
            result["pointmap"] = pts.cpu().permute((1, 2, 0))
            result["pointmap_colors"] = pts_colors.cpu().permute((1, 2, 0))
        
        return result

