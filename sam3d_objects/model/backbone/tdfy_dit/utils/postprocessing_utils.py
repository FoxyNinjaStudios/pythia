# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
import numpy as np
import torch
import utils3d
from PIL import Image
from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
import igraph
import cv2
from PIL import Image
from .random_utils import sphere_hammersley_sequence
from .render_utils import render_multiview
from ..renderers import GaussianRenderer
from ..representations import Strivec, Gaussian, MeshExtractResult
from loguru import logger

@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=32,
    resolution=128,
    num_views=500,
    debug=False,
    verbose=False,
):
    """
    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.

    Args:
        verts (torch.Tensor): Vertices of the mesh. Shape (V, 3).
        faces (torch.Tensor): Faces of the mesh. Shape (F, 3).
        max_hole_size (float): Maximum area of a hole to fill.
        resolution (int): Resolution of the rasterization.
        num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """
    # Construct cameras
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws = torch.tensor(yaws).cuda()
    pitchs = torch.tensor(pitchs).cuda()
    radius = 2.0
    fov = torch.deg2rad(torch.tensor(40)).cuda()
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    views = []
    for yaw, pitch in zip(yaws, pitchs):
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ]
            )
            .cuda()
            .float()
            * radius
        )
        view = utils3d.torch.view_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 0, 1]).float().cuda(),
        )
        views.append(view)
    views = torch.stack(views, dim=0)

    # Rasterize
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    rastctx = utils3d.torch.RastContext(backend="cuda")
    for i in tqdm(
        range(views.shape[0]),
        total=views.shape[0],
        disable=not verbose,
        desc="Rasterizing",
    ):
        view = views[i]
        buffers = utils3d.torch.rasterize_triangle_faces(
            rastctx,
            verts[None],
            faces,
            resolution,
            resolution,
            view=view,
            projection=projection,
        )
        face_id = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
        face_id = torch.unique(face_id).long()
        visblity[face_id] += 1
    visblity = visblity.float() / num_views

    # Mincut
    ## construct outer faces
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = utils3d.torch.compute_connected_components(
        faces, edges, face2edge
    )
    outer_face_indices = torch.zeros(
        faces.shape[0], dtype=torch.bool, device=faces.device
    )
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visblity[
            connected_components[i]
        ] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)

    ## construct inner faces
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if verbose:
        tqdm.write(f"Found {inner_face_indices.shape[0]} invisible faces")
    if inner_face_indices.shape[0] == 0:
        return verts, faces

    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_edges_weights = torch.norm(
        verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1
    )
    if verbose:
        tqdm.write(f"Dual graph: {dual_edges.shape[0]} edges")

    ## solve mincut problem
    ### construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    g.add_edges(dual_edges.cpu().numpy())
    g.es["weight"] = dual_edges_weights.cpu().numpy()

    ### source and target
    g.add_vertex("s")
    g.add_vertex("t")

    ### connect invisible faces to source
    g.add_edges(
        [(f, "s") for f in inner_face_indices],
        attributes={
            "weight": torch.ones(inner_face_indices.shape[0], dtype=torch.float32)
            .cpu()
            .numpy()
        },
    )

    ### connect outer faces to target
    g.add_edges(
        [(f, "t") for f in outer_face_indices],
        attributes={
            "weight": torch.ones(outer_face_indices.shape[0], dtype=torch.float32)
            .cpu()
            .numpy()
        },
    )

    ### solve mincut
    cut = g.mincut("s", "t", (np.array(g.es["weight"]) * 1000).tolist())
    remove_face_indices = torch.tensor(
        [v for v in cut.partition[0] if v < faces.shape[0]],
        dtype=torch.long,
        device=faces.device,
    )
    if verbose:
        tqdm.write(f"Mincut solved, start checking the cut")

    ### check if the cut is valid with each connected component
    to_remove_cc = utils3d.torch.compute_connected_components(
        faces[remove_face_indices]
    )
    if debug:
        tqdm.write(f"Number of connected components of the cut: {len(to_remove_cc)}")
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### check if the connected component has low visibility
        visblity_median = visblity[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f"visblity_median: {visblity_median}")
        if visblity_median > 0.25:
            continue

        #### check if the cuting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(
            face2edge[remove_face_indices[cc]], return_counts=True
        )
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[
            ~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)
        ]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(
                edges[cc_new_boundary_edge_indices]
            )
            cc_new_boundary_edges_cc_center = [
                verts[edges[cc_new_boundary_edge_indices[edge_cc]]]
                .mean(dim=1)
                .mean(dim=0)
                for edge_cc in cc_new_boundary_edge_cc
            ]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = (
                    verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]]
                    - cc_new_boundary_edges_cc_center[i]
                )
                _e2 = (
                    verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]]
                    - cc_new_boundary_edges_cc_center[i]
                )
                cc_new_boundary_edges_cc_area.append(
                    torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5
                )
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f"Area of the cutting loop: {cc_new_boundary_edges_cc_area}")
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue

        valid_remove_cc.append(cc)

    if debug:
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]
        if len(valid_remove_cc) > 0:
            vis_colors[
                remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()
            ] = [255, 0, 0]
        utils3d.io.write_ply(
            "dbg_dual.ply", face_v, edges=vis_dual_edges, vertex_colors=vis_colors
        )

        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        utils3d.io.write_ply("dbg_cut.ply", vis_verts, edges=vis_edges)

    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f"Removed {(~mask).sum()} faces by mincut")
    else:
        if verbose:
            tqdm.write(f"Removed 0 faces by mincut")

    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    verts, faces = torch.tensor(
        verts, device="cuda", dtype=torch.float32
    ), torch.tensor(faces, device="cuda", dtype=torch.int32)

    return verts, faces


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    debug: bool = False,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_hole_size (float): Maximum area of a hole to fill.
        fill_holes_max_hole_nbe (int): Maximum number of boundary edges of a hole to fill.
        fill_holes_resolution (int): Resolution of the rasterization.
        fill_holes_num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(
            f"Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
        )

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(
            vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1)
        )
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(
                f"After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
            )

    # Remove invisible faces
    # NOTE: _fill_holes uses nvdiffrast which is CUDA-only
    # Skip hole filling on MPS/CPU (will still work, just won't fill small holes)
    if fill_holes:
        if not torch.cuda.is_available():
            if verbose:
                tqdm.write(f"[MPS] Pre-processing merged mesh (Unifying {vertices.shape[0]} vertices)...")
            
            # 0. Merge coincident vertices from different chunks
            # On Apple Silicon we sometimes get micro-cracks along boundaries (or from numerical noise)
            # that show up as holes in the final mesh. Welding a bit more aggressively and cleaning
            # the mesh improves watertightness without changing the pipeline UX.
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            # trimesh API differs across versions; keep this compatible.
            # Remove degenerate faces
            try:
                nondeg = mesh.nondegenerate_faces()
                mesh.update_faces(nondeg)
            except Exception:
                pass
            # Remove duplicate faces
            try:
                uniq = mesh.unique_faces()
                mesh.update_faces(uniq)
            except Exception:
                pass
            try:
                mesh.remove_unreferenced_vertices()
            except Exception:
                pass
            # digits_vertex=3 (~0.001) is more robust for crack sealing than 4, while still preserving detail.
            mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=3)
            vertices, faces = mesh.vertices, mesh.faces
            
            if verbose:
                tqdm.write(f"[MPS] Starting hole filling (Current: {vertices.shape[0]} verts, {faces.shape[0]} faces)")
            
            # 1. Use PyVista for hole filling with a SAFER limit
            pv_mesh = pv.PolyData(
                vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1)
            )
            pv_mesh = pv_mesh.clean(inplace=False)
            # hole_size=100 is much safer than 1000. 1000 was creating "shelves" at chunk boundaries.
            filled = pv_mesh.fill_holes(100)
            
            # Convert back
            vertices, faces = filled.points, filled.faces.reshape(-1, 4)[:, 1:]

            # 2. Final pass with trimesh hole filling to close any remaining open boundaries
            # (PyVista can leave small boundary loops in some non-manifold cases).
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            try:
                nondeg = mesh.nondegenerate_faces()
                mesh.update_faces(nondeg)
            except Exception:
                pass
            try:
                uniq = mesh.unique_faces()
                mesh.update_faces(uniq)
            except Exception:
                pass
            try:
                mesh.remove_unreferenced_vertices()
            except Exception:
                pass
            mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=3)
            try:
                trimesh.repair.fill_holes(mesh)
            except Exception:
                # Best-effort: keep the filled result even if repair fails.
                pass
            vertices, faces = mesh.vertices, mesh.faces
            
            if verbose:
                tqdm.write(f"[MPS] Completed hole filling: {vertices.shape[0]} verts, {faces.shape[0]} faces")
        else:
            vertices, faces = (
                torch.tensor(vertices).cuda(),
                torch.tensor(faces.astype(np.int32)).cuda(),
            )
            vertices, faces = _fill_holes(
                vertices,
                faces,
                max_hole_size=fill_holes_max_hole_size,
                max_hole_nbe=fill_holes_max_hole_nbe,
                resolution=fill_holes_resolution,
                num_views=fill_holes_num_views,
                debug=debug,
                verbose=verbose,
            )
            vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
            if verbose:
                tqdm.write(
                    f"After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
                )

    # Keep significant connected components (Remove tiny floating debris)
    if vertices.shape[0] > 0:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Sort by vertex count
            components = sorted(components, key=lambda x: len(x.vertices), reverse=True)
            largest_size = len(components[0].vertices)
            # Keep parts that are at least 10% of the largest component (main body)
            keep_components = [c for c in components if len(c.vertices) > largest_size * 0.1]
            
            if len(keep_components) < len(components):
                mesh = trimesh.util.concatenate(keep_components)
                vertices, faces = mesh.vertices, mesh.faces
                if verbose:
                    tqdm.write(
                        f"After keeping {len(keep_components)} significant components: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
                    )

    return vertices, faces


def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs

@torch.inference_mode(False)
@torch.enable_grad()
def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal["fast", "opt"] = "opt",
    lambda_tv: float = 1e-2,
    verbose: bool = False,
    rendering_engine: str = "nvdiffrast",  # nvdiffrast OR "pytorch3d"
    device: str = "cuda",

):
    """
    Bake texture to a mesh from multiple observations.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        texture_size (int): Size of the texture.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of texture baking.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
    """


    vertices = torch.tensor(vertices).to(device)
    faces = torch.tensor(faces.astype(np.int32)).to(device)
    uvs = torch.tensor(uvs).to(device)
    observations = [torch.tensor(obs / 255.0).float().to(device) for obs in observations]
    masks = [torch.tensor(m > 0).bool().to(device) for m in masks]
    views = [
        utils3d.torch.extrinsics_to_view(torch.tensor(extr).to(device))
        for extr in extrinsics
    ]
    projections = [
        utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).to(device), near, far)
        for intr in intrinsics
    ]

    if mode == "fast":
        texture = torch.zeros(
            (texture_size * texture_size, 3), dtype=torch.float32
        ).to(device)
        texture_weights = torch.zeros(
            (texture_size * texture_size), dtype=torch.float32
        ).to(device)
        rastctx = utils3d.torch.RastContext(backend=device if device.startswith("cuda") else "cuda")
        for observation, view, projection in tqdm(
            zip(observations, views, projections),
            total=len(observations),
            disable=not verbose,
            desc="Texture baking (fast)",
        ):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx,
                    vertices[None],
                    faces,
                    observation.shape[1],
                    observation.shape[0],
                    uv=uvs[None],
                    view=view,
                    projection=projection,
                )
                uv_map = rast["uv"][0].detach().flip(0)
                mask = rast["mask"][0].detach().bool() & masks[0]

            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(
                0,
                idx,
                torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device),
            )

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(
            texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)

        # inpaint
        mask = (
            (texture_weights == 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
            .reshape(texture_size, texture_size)
        )
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    elif mode == "opt":
        rastctx = utils3d.torch.RastContext(backend=device if device.startswith("cuda") else "cuda")
        observations = [observations.flip(0) for observations in observations]
        masks = [m.flip(0) for m in masks]
        _uv = []
        _uv_dr = []
        for observation, view, projection in tqdm(
            zip(observations, views, projections),
            total=len(views),
            disable=not verbose,
            desc="Texture baking (opt): UV",
        ):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx,
                    vertices[None],
                    faces,
                    observation.shape[1],
                    observation.shape[0],
                    uv=uvs[None],
                    view=view,
                    projection=projection,
                )
                _uv.append(rast["uv"].detach())
                _uv_dr.append(rast["uv_dr"].detach())

        texture = torch.nn.Parameter(
            torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).to(device)
        )
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (
                1 + np.cos(np.pi * step / total_steps)
            )

        def tv_loss(texture):
            return torch.nn.functional.l1_loss(
                texture[:, :-1, :, :], texture[:, 1:, :, :]
            ) + torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])



        def render_pt3d_texture(texture, uv, uv_dr=None):
            import torch.nn.functional as F
            texture_perm = texture.permute(0, 3, 1, 2)
            grid = uv * 2 - 1
            if grid.dim() == 3:
                grid = grid.unsqueeze(0)  # (1, H, W, 2)
            elif grid.dim() == 4 and grid.shape[0] == 1:
                pass  
            elif grid.dim() == 4 and grid.shape[1] == 1:
                grid = grid.squeeze(1)  # remove extra batch dimension if necessary
            else:
                raise ValueError(f"Unexpected grid shape: {grid.shape}")
            render = F.grid_sample(
                texture_perm, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            render = render.permute(0, 2, 3, 1)[0]  # (H_out, W_out, 3)
            return render
        
        
        total_steps = 2500
        
        with tqdm(
            total=total_steps,
            disable=not verbose,
            desc="Texture baking (opt): optimizing",
            ) as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                selected = np.random.randint(0, len(views))
                uv, uv_dr, observation, mask = (
                    _uv[selected],
                    _uv_dr[selected],
                    observations[selected],
                    masks[selected],
                )
                
                if rendering_engine == "nvdiffrast":
                    import nvdiffrast.torch as dr
                    render = dr.texture(texture, uv, uv_dr)[0]

                if rendering_engine == "pytorch3d":
                    render = render_pt3d_texture(texture, uv)
                    
                loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)
                loss.backward()
                optimizer.step()
                # annealing
                optimizer.param_groups[0]["lr"] = cosine_anealing(
                    optimizer, step, total_steps, 1e-2, 1e-5
                    )
                pbar.set_postfix({"loss": loss.item()})
                pbar.update()
        texture = np.clip(
            texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)
        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
        )["mask"][0].detach().cpu().numpy().astype(np.uint8)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return texture


def _resample_vertex_colors(orig_vertices, orig_colors, new_vertices):
    """Transfer per-vertex RGB onto a possibly decimated / re-indexed vertex set.

    ``postprocess_mesh`` simplifies and re-indexes vertices, so colors indexed to
    the *original* vertex set no longer align. We transfer by nearest original
    vertex position (albedo is locally smooth, so nearest-neighbor is clean). When
    the vertex set is unchanged (e.g. simplify=0, no holes filled) this is an
    identity map, so byte-for-byte behavior is preserved on that path.

    Args:
        orig_vertices (np.array): original vertices, shape (V0, 3).
        orig_colors (np.array): original per-vertex color in [0, 1] or [0, 255], shape (V0, >=3).
        new_vertices (np.array): postprocessed vertices, shape (V, 3).

    Returns:
        np.uint8 array of shape (V, 4) RGBA suitable for ``trimesh`` vertex_colors.
    """
    from scipy.spatial import cKDTree

    colors = np.asarray(orig_colors, dtype=np.float32)[:, :3]
    # FlexiCubes emits sigmoid colors in [0, 1]; tolerate a [0, 255] source too.
    if colors.size and colors.max() > 1.0 + 1e-3:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)

    if new_vertices.shape[0] == orig_vertices.shape[0] and np.array_equal(
        new_vertices, orig_vertices
    ):
        idx = np.arange(new_vertices.shape[0])
    else:
        _, idx = cKDTree(orig_vertices).query(new_vertices, k=1)

    rgb = (colors[idx] * 255.0).round().astype(np.uint8)
    alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


# Spherical-harmonics DC → RGB constant (SH band 0). color = 0.5 + SH_C0 * f_dc
SH_C0 = 0.28209479177387814


def rasterize_uv_barycentric(uvs: np.array, faces: np.array, texture_size: int):
    """Rasterize a UV atlas into per-texel (face id, barycentric weights).

    A pure-numpy software rasterizer — no CUDA / nvdiffrast — used to bake a
    texture on Apple Silicon. UV convention matches glTF/trimesh: u→column,
    v→row with the image top at v=1.

    Args:
        uvs (np.array): per-vertex UVs in [0, 1], shape (V, 2).
        faces (np.array): triangle vertex indices, shape (F, 3).
        texture_size (int): output texture edge length S.

    Returns:
        face_id (np.int32, (S, S)): covering face per texel, -1 where empty.
        bary (np.float32, (S, S, 3)): barycentric weights per texel.
    """
    S = int(texture_size)
    face_id = np.full((S, S), -1, dtype=np.int32)
    bary = np.zeros((S, S, 3), dtype=np.float32)

    # UV → pixel coords (x = column, y = row).
    px = np.empty_like(uvs, dtype=np.float64)
    px[:, 0] = uvs[:, 0] * (S - 1)
    px[:, 1] = (1.0 - uvs[:, 1]) * (S - 1)
    tri = px[faces]  # (F, 3, 2)

    for f in range(faces.shape[0]):
        p0, p1, p2 = tri[f]
        minx = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
        maxx = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), S - 1)
        miny = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
        maxy = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), S - 1)
        if minx > maxx or miny > maxy:
            continue
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(denom) < 1e-12:
            continue
        gx, gy = np.meshgrid(
            np.arange(minx, maxx + 1), np.arange(miny, maxy + 1)
        )
        w0 = ((p1[1] - p2[1]) * (gx - p2[0]) + (p2[0] - p1[0]) * (gy - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (gx - p2[0]) + (p0[0] - p2[0]) * (gy - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        rows = gy[inside]
        cols = gx[inside]
        face_id[rows, cols] = f
        bary[rows, cols, 0] = w0[inside]
        bary[rows, cols, 1] = w1[inside]
        bary[rows, cols, 2] = w2[inside]

    return face_id, bary


def _gaussian_xyz_colors(gaussian: Gaussian):
    """Return (xyz (G,3), rgb (G,3) in [0,1]) from a Gaussian's SH-DC color."""
    xyz = gaussian.get_xyz.detach().float().cpu().numpy()
    dc = gaussian.get_features.detach().float().cpu().numpy()
    dc = dc.reshape(dc.shape[0], -1)[:, :3]
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    return xyz, rgb


def _vertex_colors_from_gaussian(new_vertices, gaussian: Gaussian, k: int = 8):
    """Per-vertex RGBA sampled robustly from the Gaussian appearance field.

    The mesh decoder's own vertex color head is low-fidelity and tends toward a
    washed-out neutral gray. The Gaussian's SH-DC colors carry the true, more
    saturated appearance, so we color each mesh vertex from nearby Gaussians.
    Because trimesh Gouraud-interpolates vertex colors across faces, the result is
    smooth (unlike the per-texel nearest bake, which speckles).

    Silhouette/edge vertices sit where Gaussians are sparse, so their single
    nearest Gaussian is often an outlier (a stray dark/red splat), which shows up
    as colored speckle along the mesh boundary. Taking a per-channel median over
    the ``k`` nearest Gaussians rejects those isolated outliers while keeping the
    interior colors intact, cleaning up the edges.

    Args:
        new_vertices (np.array): mesh vertices in the Gaussian's frame, shape (V, 3).
        gaussian (Gaussian): appearance source.
        k (int): number of nearest Gaussians to aggregate (median). k=1 = plain nearest.

    Returns:
        np.uint8 array of shape (V, 4) RGBA for ``trimesh`` vertex_colors.
    """
    from scipy.spatial import cKDTree

    gxyz, grgb = _gaussian_xyz_colors(gaussian)
    k = int(max(1, min(k, gxyz.shape[0])))
    _, idx = cKDTree(gxyz).query(new_vertices, k=k)
    if k == 1:
        rgb = grgb[idx]
    else:
        # (V, k, 3) neighbor colors -> per-channel median rejects single outliers.
        rgb = np.median(grgb[idx], axis=1)
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


def _smooth_color_outliers(faces, colors, iterations: int = 3, threshold: float = 38.0):
    """Clean isolated per-vertex color speckle (mainly silhouette/edge splats).

    For each vertex we compute the mean color of its 1-ring mesh neighbors. Only
    vertices whose color deviates from that neighborhood mean by more than
    ``threshold`` (i.e. outliers/specks) are pulled to the mean; smooth interior
    vertices are left untouched, so real texture detail is preserved while stray
    dark/red edge splats are blended into the surrounding surface.

    Args:
        faces (np.array): triangle vertex indices, shape (F, 3).
        colors (np.array): per-vertex RGBA uint8, shape (V, 4).
        iterations (int): smoothing passes.
        threshold (float): L2 color distance (0-255) above which a vertex is an outlier.

    Returns:
        np.uint8 RGBA array, shape (V, 4).
    """
    from scipy.sparse import coo_matrix

    V = colors.shape[0]
    if V == 0 or faces.size == 0:
        return colors

    rgb = colors[:, :3].astype(np.float32)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    adj = coo_matrix((np.ones(i.shape[0], np.float32), (i, j)), shape=(V, V)).tocsr()
    adj.data[:] = 1.0  # dedupe: unit weight per unique neighbor
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0

    for _ in range(int(iterations)):
        nbr_mean = (adj @ rgb) / deg[:, None]
        dev = np.linalg.norm(rgb - nbr_mean, axis=1)
        out = dev > threshold
        if not out.any():
            break
        rgb[out] = nbr_mean[out]

    out_rgb = np.clip(rgb, 0.0, 255.0).round().astype(np.uint8)
    return np.concatenate([out_rgb, colors[:, 3:4]], axis=1)


def bake_texture_portable(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    texture_size: int = 1024,
    vert_colors: Optional[np.array] = None,
    gaussian: Optional[Gaussian] = None,
    inpaint: bool = True,
    verbose: bool = False,
) -> np.array:
    """Bake a UV texture on Apple Silicon without any CUDA rasterizer.

    Two color sources:
      * ``vert_colors`` (Phase 2): barycentric interpolation of per-vertex RGB.
      * ``gaussian`` (Phase 3): each covered texel maps to a 3D surface point;
        color = nearest Gaussian's SH-DC color (denser than mesh vertices).

    Args:
        vertices (np.array): mesh vertices, shape (V, 3), in the same frame as ``gaussian``.
        faces (np.array): triangles, shape (F, 3).
        uvs (np.array): per-vertex UVs in [0, 1], shape (V, 2).
        texture_size (int): texture edge length.
        vert_colors (np.array | None): per-vertex RGB in [0, 1], shape (V, 3).
        gaussian (Gaussian | None): appearance source for the nearest-neighbor bake.
        inpaint (bool): fill uncovered texels (atlas seams) with cv2 Telea.

    Returns:
        np.uint8 texture, shape (S, S, 3).
    """
    from scipy.spatial import cKDTree

    S = int(texture_size)
    if verbose:
        logger.info(f"[portable-bake] rasterizing {faces.shape[0]} faces into {S}x{S} atlas")
    face_id, bary = rasterize_uv_barycentric(uvs, faces, S)
    covered = face_id >= 0
    tex = np.zeros((S, S, 3), dtype=np.float32)

    if covered.any():
        fids = face_id[covered]
        b = bary[covered][:, :, None]          # (M, 3, 1)
        tri_v = faces[fids]                     # (M, 3) vertex indices

        if gaussian is not None:
            pts = (vertices[tri_v] * b).sum(axis=1)        # (M, 3) surface points
            gxyz, grgb = _gaussian_xyz_colors(gaussian)
            _, idx = cKDTree(gxyz).query(pts, k=1)
            colors = grgb[idx]
        else:
            vc = np.asarray(vert_colors, dtype=np.float32)[:, :3]
            if vc.size and vc.max() > 1.0 + 1e-3:
                vc = vc / 255.0
            vc = np.clip(vc, 0.0, 1.0)
            colors = (vc[tri_v] * b).sum(axis=1)           # (M, 3)

        tex[covered] = colors

    tex = np.clip(tex * 255.0, 0, 255).astype(np.uint8)
    if inpaint:
        mask = (~covered).astype(np.uint8)
        tex = cv2.inpaint(tex, mask, 3, cv2.INPAINT_TELEA)
    return tex


def to_glb(
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    debug: bool = False,
    verbose: bool = True,
    with_mesh_postprocess=True,
    with_texture_baking=True,
    use_vertex_color=False,
    rendering_engine: str = "nvdiffrast",  # nvdiffrast OR "pytorch3d"
    bake_backend: str = "render",  # "render" (CUDA multi-view) OR "portable" (CPU/MPS)
    bake_source: str = "gaussian",  # portable bake color source: "gaussian" OR "vertex"
    mesh_smooth_iterations: int = 5,  # Taubin passes to de-staircase the silhouette (0 = off)
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_size (float): Maximum area of a hole to fill.
        texture_size (int): Size of the texture.
        debug (bool): Whether to print debug information.
        verbose (bool): Whether to print progress.
    """
    vertices = mesh.vertices.float().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    _has_vert_colors = (
        getattr(mesh, "vertex_attrs", None) is not None
        and mesh.vertex_attrs.shape[0] == vertices.shape[0]
        and mesh.vertex_attrs.shape[-1] >= 3
    )
    vert_colors = mesh.vertex_attrs[:, :3].float().cpu().numpy() if _has_vert_colors else None
    orig_vertices = vertices.copy()

    if with_mesh_postprocess:
        # mesh postprocess
        # ensure simplify is in [0, 0.999] for sqrt safety
        safe_simplify = min(max(simplify, 0.0), 0.999)
        vertices, faces = postprocess_mesh(
            vertices,
            faces,
            simplify=simplify > 0,
            simplify_ratio=simplify,
            fill_holes=fill_holes,
            fill_holes_max_hole_size=fill_holes_max_size,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - safe_simplify)),
            fill_holes_resolution=1024,
            fill_holes_num_views=1000,
            debug=debug,
            verbose=verbose,
        )

    # Phase 1: carry per-vertex color through the postprocess vertex remap so the
    # exported GLB is colored instead of flat gray. Only when we are not baking a
    # UV texture (which supersedes vertex color). Prefer the Gaussian appearance
    # field when available (its SH-DC colors are far more saturated than the mesh
    # decoder's near-gray vertex head); fall back to the decoder colors otherwise.
    resampled_colors = None
    if use_vertex_color and not with_texture_baking and vertices.size != 0:
        if isinstance(app_rep, Gaussian):
            resampled_colors = _vertex_colors_from_gaussian(vertices, app_rep)
        elif vert_colors is not None:
            resampled_colors = _resample_vertex_colors(orig_vertices, vert_colors, vertices)

    if with_texture_baking:
        # parametrize mesh (xatlas; CPU, portable)
        vertices, faces, uvs = parametrize_mesh(vertices, faces)
        logger.info(f"Baking texture (backend={bake_backend}, source={bake_source}) ...")

        if bake_backend == "portable":
            # Apple-Silicon path: no CUDA rasterizer, no multi-view render.
            if bake_source == "gaussian" and isinstance(app_rep, Gaussian):
                texture = bake_texture_portable(
                    vertices, faces, uvs, texture_size,
                    gaussian=app_rep, verbose=verbose,
                )
            else:
                # Phase 2: bake per-vertex color. Colors are indexed to the
                # original vertex set; transfer them onto the parametrized
                # vertices by position first.
                vc = None
                if vert_colors is not None:
                    vc = _resample_vertex_colors(orig_vertices, vert_colors, vertices)[:, :3].astype(np.float32) / 255.0
                texture = bake_texture_portable(
                    vertices, faces, uvs, texture_size,
                    vert_colors=vc, verbose=verbose,
                )
        else:
            # Upstream CUDA path: render the appearance rep from many views and
            # optimize a texture to match.
            observations, extrinsics, intrinsics = render_multiview(
                app_rep, resolution=1024, nviews=100
            )
            masks = [np.any(observation > 0, axis=-1) for observation in observations]
            extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
            intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
            texture = bake_texture(
                vertices,
                faces,
                uvs,
                observations,
                masks,
                extrinsics,
                intrinsics,
                texture_size=texture_size,
                mode="opt",
                lambda_tv=0.01,
                verbose=verbose,
                rendering_engine=rendering_engine
            )
        texture = Image.fromarray(texture)
        material = trimesh.visual.material.PBRMaterial(
            roughnessFactor=1.0,
            baseColorTexture=texture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        )

    # Some repair/fill steps can create NaN/Inf vertices on CPU (rare, but catastrophic for export).
    # Filter invalid vertices and drop faces that reference them.
    if vertices.size != 0:
        # Expected coordinates are normalized (roughly within [-1, 1]).
        # Treat extreme outliers as invalid to avoid numeric overflow during export.
        valid_v = np.isfinite(vertices).all(axis=1) & (np.abs(vertices).max(axis=1) < 10.0)
        if not np.all(valid_v):
            keep_faces = valid_v[faces].all(axis=1)
            faces = faces[keep_faces]
            remap = -np.ones(vertices.shape[0], dtype=np.int64)
            remap[np.where(valid_v)[0]] = np.arange(int(valid_v.sum()), dtype=np.int64)
            faces = remap[faces]
            vertices = vertices[valid_v]
            if resampled_colors is not None:
                resampled_colors = resampled_colors[valid_v]

    # rotate mesh (from z-up to y-up)
    # Avoid numpy matmul here: BLAS implementations can emit divide-by-zero/overflow warnings
    # when input contains pathological values. Component-wise rotation is robust.
    if vertices.size != 0:
        x = vertices[:, 0].copy()
        y = vertices[:, 1].copy()
        z = vertices[:, 2].copy()
        vertices[:, 0] = x
        vertices[:, 1] = z
        vertices[:, 2] = -y

    # Edge cleanup: blend isolated color specks (stray dark/red splats along the
    # silhouette) into the surrounding surface, leaving smooth regions intact.
    if resampled_colors is not None and faces.size != 0:
        resampled_colors = _smooth_color_outliers(faces, resampled_colors)

    if with_texture_baking:
        mesh = trimesh.Trimesh(
            vertices,
            faces,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        )
    elif resampled_colors is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = resampled_colors
    else:
        mesh = trimesh.Trimesh(vertices, faces)

    # Silhouette de-fur: the mesh is extracted from a 64^3 sparse voxel grid
    # via flexicubes, which leaves isolated "spike" vertices shooting out of the
    # silhouette (the "fur" along backrest/leg edges). Whole-mesh smoothers make
    # this WORSE — Taubin's volume-preserving inverse step amplifies the spikes
    # (measured ~4x more outliers), and Laplacian NaNs on non-watertight meshes.
    # Instead we despike: pull only the outlier vertices onto their neighbor
    # centroid. It only moves vertex positions, so UVs and per-vertex colors
    # (both indexed by vertex id) stay valid.
    if mesh_smooth_iterations > 0 and mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0:
        mesh = _smooth_mesh_surface(mesh, iterations=mesh_smooth_iterations, verbose=verbose)

    return mesh


def _smooth_mesh_surface(mesh: trimesh.Trimesh, iterations: int = 5, verbose: bool = False) -> trimesh.Trimesh:
    """Remove silhouette "fur" spikes by clamping outlier vertices.

    The flexicubes extraction on the 64^3 SLAT grid leaves isolated vertices
    that shoot out of the surface (spikes / "fur" along the silhouette). Rather
    than a global low-pass filter (Taubin/Laplacian actually *amplify* these on
    the noisy, non-watertight decoder mesh), we detect vertices whose distance to
    the centroid of their 1-ring neighbors exceeds a small fraction of the mesh
    bounding box and snap just those onto the neighbor centroid, iterating a few
    times. Interior/true-detail vertices are untouched, and connectivity is
    preserved so UV coordinates and per-vertex colors remain valid. Failures are
    non-fatal - the original mesh is returned.
    """
    try:
        import scipy.sparse as sp

        V = np.asarray(mesh.vertices, dtype=np.float64)
        edges = np.asarray(mesh.edges_unique)
        if V.shape[0] == 0 or edges.shape[0] == 0:
            return mesh

        n = V.shape[0]
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        W = sp.csr_matrix((np.ones(rows.shape[0], np.float64), (rows, cols)), shape=(n, n))
        deg = np.asarray(W.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0

        scale = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) or 1.0
        thr = 0.01 * scale  # a single vertex >1% of the bbox from its neighbors is a spike

        moved = 0
        # A couple of extra passes beyond `iterations` lets clusters of adjacent
        # spikes settle (their neighbors may themselves be spikes on pass 1).
        for _ in range(max(1, int(iterations)) + 3):
            nbr = (W @ V) / deg[:, None]
            d = np.linalg.norm(V - nbr, axis=1)
            out = d > thr
            k = int(out.sum())
            if k == 0:
                break
            V[out] = nbr[out]
            moved += k

        mesh.vertices = V
        if verbose:
            logger.info(f"Despiked mesh silhouette ({moved} outlier vertex moves)")
    except Exception as exc:  # pragma: no cover - despiking is best-effort
        logger.warning(f"Mesh silhouette despike skipped: {exc}")
    return mesh


def simplify_gs(
    gs: Gaussian,
    simplify: float = 0.95,
    verbose: bool = True,
):
    """
    Simplify 3D Gaussians
    NOTE: this function is not used in the current implementation for the unsatisfactory performance.

    Args:
        gs (Gaussian): 3D Gaussian.
        simplify (float): Ratio of Gaussians to remove in simplification.
    """
    if simplify <= 0:
        return gs

    # simplify
    observations, extrinsics, intrinsics = render_multiview(
        gs, resolution=1024, nviews=100
    )
    observations = [
        torch.tensor(obs / 255.0).float().cuda().permute(2, 0, 1)
        for obs in observations
    ]

    # Following https://arxiv.org/pdf/2411.06019
    renderer = GaussianRenderer(
        {
            "resolution": 1024,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0, 0, 0),
        }
    )
    new_gs = Gaussian(**gs.init_params)
    new_gs._features_dc = gs._features_dc.clone()
    new_gs._features_rest = (
        gs._features_rest.clone() if gs._features_rest is not None else None
    )
    new_gs._opacity = torch.nn.Parameter(gs._opacity.clone())
    new_gs._rotation = torch.nn.Parameter(gs._rotation.clone())
    new_gs._scaling = torch.nn.Parameter(gs._scaling.clone())
    new_gs._xyz = torch.nn.Parameter(gs._xyz.clone())

    start_lr = [1e-4, 1e-3, 5e-3, 0.025]
    end_lr = [1e-6, 1e-5, 5e-5, 0.00025]
    optimizer = torch.optim.Adam(
        [
            {"params": new_gs._xyz, "lr": start_lr[0]},
            {"params": new_gs._rotation, "lr": start_lr[1]},
            {"params": new_gs._scaling, "lr": start_lr[2]},
            {"params": new_gs._opacity, "lr": start_lr[3]},
        ],
        lr=start_lr[0],
    )

    def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return start_lr * (end_lr / start_lr) ** (step / total_steps)

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return end_lr + 0.5 * (start_lr - end_lr) * (
            1 + np.cos(np.pi * step / total_steps)
        )

    _zeta = new_gs.get_opacity.clone().detach().squeeze()
    _lambda = torch.zeros_like(_zeta)
    _delta = 1e-7
    _interval = 10
    num_target = int((1 - simplify) * _zeta.shape[0])

    with tqdm(total=2500, disable=not verbose, desc="Simplifying Gaussian") as pbar:
        for i in range(2500):
            # prune
            if i % 100 == 0:
                mask = new_gs.get_opacity.squeeze() > 0.05
                mask = torch.nonzero(mask).squeeze()
                new_gs._xyz = torch.nn.Parameter(new_gs._xyz[mask])
                new_gs._rotation = torch.nn.Parameter(new_gs._rotation[mask])
                new_gs._scaling = torch.nn.Parameter(new_gs._scaling[mask])
                new_gs._opacity = torch.nn.Parameter(new_gs._opacity[mask])
                new_gs._features_dc = new_gs._features_dc[mask]
                new_gs._features_rest = (
                    new_gs._features_rest[mask]
                    if new_gs._features_rest is not None
                    else None
                )
                _zeta = _zeta[mask]
                _lambda = _lambda[mask]
                # update optimizer state
                for param_group, new_param in zip(
                    optimizer.param_groups,
                    [new_gs._xyz, new_gs._rotation, new_gs._scaling, new_gs._opacity],
                ):
                    stored_state = optimizer.state[param_group["params"][0]]
                    if "exp_avg" in stored_state:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    del optimizer.state[param_group["params"][0]]
                    param_group["params"][0] = new_param
                    optimizer.state[param_group["params"][0]] = stored_state

            opacity = new_gs.get_opacity.squeeze()

            # sparisfy
            if i % _interval == 0:
                _zeta = _lambda + opacity.detach()
                if opacity.shape[0] > num_target:
                    index = _zeta.topk(num_target)[1]
                    _m = torch.ones_like(_zeta, dtype=torch.bool)
                    _m[index] = 0
                    _zeta[_m] = 0
                _lambda = _lambda + opacity.detach() - _zeta

            # sample a random view
            view_idx = np.random.randint(len(observations))
            observation = observations[view_idx]
            extrinsic = extrinsics[view_idx]
            intrinsic = intrinsics[view_idx]

            color = renderer.render(new_gs, extrinsic, intrinsic)["color"]
            rgb_loss = torch.nn.functional.l1_loss(color, observation)
            loss = rgb_loss + _delta * torch.sum(
                torch.pow(_lambda + opacity - _zeta, 2)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update lr
            for j in range(len(optimizer.param_groups)):
                optimizer.param_groups[j]["lr"] = cosine_anealing(
                    optimizer, i, 2500, start_lr[j], end_lr[j]
                )

            pbar.set_postfix(
                {
                    "loss": rgb_loss.item(),
                    "num": opacity.shape[0],
                    "lambda": _lambda.mean().item(),
                }
            )
            pbar.update()

    new_gs._xyz = new_gs._xyz.data
    new_gs._rotation = new_gs._rotation.data
    new_gs._scaling = new_gs._scaling.data
    new_gs._opacity = new_gs._opacity.data

    return new_gs
