# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes


class MeshExtractResult:
    def __init__(self, vertices, faces, vertex_attrs=None, res=64):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = vertices.shape[0] != 0 and faces.shape[0] != 0

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None

    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # print(face_normals.min(), face_normals.max(), face_normals.shape)
        return face_normals[:, None, :].repeat(1, 3, 1)

    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals


class SparseFeatures2Mesh:
    def __init__(self, device="cpu", res=64, use_color=True):  # Changed from cuda for CPU compatibility
        """
        a model to generate a mesh from sparse features structures using flexicube
        """
        super().__init__()
        self.device = device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()

    def _calc_layout(self):
        LAYOUTS = {
            "sdf": {"shape": (8, 1), "size": 8},
            "deform": {"shape": (8, 3), "size": 8 * 3},
            "weights": {"shape": (21,), "size": 21},
        }
        if self.use_color:
            """
            6 channel color including normal map
            """
            LAYOUTS["color"] = {
                "shape": (
                    8,
                    6,
                ),
                "size": 8 * 6,
            }
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.feats_channels = start

    def get_layout(self, feats: torch.Tensor, name: str):
        if name not in self.layouts:
            return None
        return feats[
            :, self.layouts[name]["range"][0] : self.layouts[name]["range"][1]
        ].reshape(-1, *self.layouts[name]["shape"])

    def __call__(self, cubefeats: SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Currently uses a single-pass extraction for maximum detail and consistency.
        The underlying streaming conv/attn layers ensure we stay under memory limits.
        """
        return self._single_call(cubefeats, training=training)
    
    def _single_call(self, cubefeats: SparseTensor, training=False):
        """Original single-pass mesh extraction (for small grids)."""
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats

        sdf, deform, color, weights = [
            self.get_layout(feats, name)
            for name in ["sdf", "deform", "color", "weights"]
        ]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(
            coords, torch.cat(v_attrs, dim=-1), training=training
        )
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res + 1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        if self.use_color:
            sdf_d, deform_d, colors_d = (
                v_attrs_d[..., 0],
                v_attrs_d[..., 1:4],
                v_attrs_d[..., 4:],
            )
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        # Ensure internal grid templates are on the same device as input features
        # This is critical when falling back to CPU for triangulation stability.
        if self.reg_c.device != feats.device:
            self.reg_c = self.reg_c.to(feats.device)
        if self.reg_v.device != feats.device:
            self.reg_v = self.reg_v.to(device=feats.device, dtype=feats.dtype)
            
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
        # Stability: Run triangulation on the same device as features (now CPU)
        # to avoid MPS-specific index_add_ bugs.
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training,
        )

        mesh = MeshExtractResult(
            vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res
        )
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:, :20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        return mesh
    
    def _chunked_call(self, cubefeats: SparseTensor, training=False, n_chunks=4):
        """
        Chunked mesh extraction for large grids.
        
        Splits the grid along Z-axis into n_chunks and processes each separately.
        Each chunk uses ~1/n_chunks of the dense grid memory.
        """
        import gc
        
        coords = cubefeats.coords[:, 1:]  # Remove batch dim
        feats = cubefeats.feats
        device = feats.device
        
        # Extract all features
        sdf, deform, color, weights = [
            self.get_layout(feats, name)
            for name in ["sdf", "deform", "color", "weights"]
        ]
        sdf += self.sdf_bias
        v_attrs_list = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_attrs_cat = torch.cat(v_attrs_list, dim=-1)
        
        # Convert sparse cubes to sparse vertices
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, v_attrs_cat, training=training)
        
        # Split along Z-axis
        z_coords = coords[:, 2]
        z_min, z_max = z_coords.min().item(), z_coords.max().item()
        chunk_size = (z_max - z_min + 1) / n_chunks
        
        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0
        
        for chunk_idx in range(n_chunks):
            z_start = int(z_min + chunk_idx * chunk_size)
            z_end = int(z_min + (chunk_idx + 1) * chunk_size) if chunk_idx < n_chunks - 1 else int(z_max + 1)
            
            # Overlap by 1 to ensure surface continuity
            z_start_padded = max(0, z_start - 1)
            z_end_padded = min(self.res, z_end + 1)
            chunk_res_z = z_end_padded - z_start_padded
            
            # Filter voxels in this chunk (using cube coords)
            chunk_mask_cubes = (coords[:, 2] >= z_start_padded) & (coords[:, 2] < z_end_padded)
            chunk_coords = coords[chunk_mask_cubes].clone()
            chunk_weights = weights[chunk_mask_cubes]
            
            # Filter vertices in this chunk
            chunk_mask_verts = (v_pos[:, 2] >= z_start_padded) & (v_pos[:, 2] <= z_end_padded)
            chunk_v_pos = v_pos[chunk_mask_verts].clone()
            chunk_v_attrs = v_attrs[chunk_mask_verts]
            
            if chunk_coords.shape[0] == 0 or chunk_v_pos.shape[0] == 0:
                continue
            
            # Shift Z coordinates to local chunk space
            chunk_coords[:, 2] -= z_start_padded
            chunk_v_pos[:, 2] -= z_start_padded
            
            # Create dense grid for this chunk only (much smaller!)
            chunk_v_attrs_d = get_dense_attrs(
                chunk_v_pos, chunk_v_attrs, 
                res=self.res + 1,  # X,Y full resolution, but only chunk_res_z in Z (handled by coords)
                sdf_init=True
            ).reshape(self.res + 1, self.res + 1, self.res + 1, -1)[:, :, :chunk_res_z + 1, :].reshape(-1, chunk_v_attrs.shape[-1])
            
            chunk_weights_d = get_dense_attrs(
                chunk_coords, chunk_weights,
                res=self.res,
                sdf_init=False
            ).reshape(self.res, self.res, self.res, -1)[:, :, :chunk_res_z, :].reshape(-1, chunk_weights.shape[-1])
            
            if self.use_color:
                sdf_d = chunk_v_attrs_d[..., 0]
                deform_d = chunk_v_attrs_d[..., 1:4]
                colors_d = chunk_v_attrs_d[..., 4:]
            else:
                sdf_d = chunk_v_attrs_d[..., 0]
                deform_d = chunk_v_attrs_d[..., 1:4]
                colors_d = None
            
            # Build chunk-local grid
            chunk_verts, chunk_cube = construct_dense_grid(res=self.res, device=device)
            # Select only the Z-slice we need
            chunk_verts = chunk_verts.reshape(self.res + 1, self.res + 1, self.res + 1, 3)[:, :, :chunk_res_z + 1, :].reshape(-1, 3)
            chunk_cube = chunk_cube.reshape(self.res, self.res, self.res, 8)[:, :, :chunk_res_z, :].reshape(-1, 8)
            
            # Adjust cube indices for chunk vertex layout
            # This requires recomputing cube_idx based on chunk dimensions
            res_v_z = chunk_res_z + 1
            res_v_xy = self.res + 1
            cube_corners = torch.tensor(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], 
                 [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], 
                dtype=torch.int, device=device
            )
            vertsid = torch.arange(res_v_xy * res_v_xy * res_v_z, device=device)
            coordsid = vertsid.reshape(res_v_xy, res_v_xy, res_v_z)[:self.res, :self.res, :chunk_res_z].flatten()
            cube_corners_bias = (cube_corners[:, 0] * res_v_xy + cube_corners[:, 1]) * res_v_z + cube_corners[:, 2]
            chunk_cube = coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0)
            chunk_verts = torch.stack([
                vertsid // (res_v_xy * res_v_z),
                (vertsid // res_v_z) % res_v_xy,
                vertsid % res_v_z
            ], dim=1).float()
            
            x_nx3 = get_defomed_verts(chunk_verts, deform_d, self.res)
            
            try:
                vertices, faces, L_dev, colors = self.mesh_extractor(
                    voxelgrid_vertices=x_nx3,
                    scalar_field=sdf_d,
                    cube_idx=chunk_cube,
                    resolution=[self.res, self.res, chunk_res_z],  # Non-uniform resolution
                    beta=chunk_weights_d[:, :12],
                    alpha=chunk_weights_d[:, 12:20],
                    gamma_f=chunk_weights_d[:, 20],
                    voxelgrid_colors=colors_d,
                    training=training,
                )
            except Exception as e:
                print(f"[CHUNK {chunk_idx}] FlexiCubes failed: {e}")
                continue
            
            if vertices.shape[0] > 0:
                # Restore Z coordinate offset
                vertices[:, 2] += z_start_padded / self.res - 0.5  # Convert to normalized coords
                
                # Offset faces by current vertex count
                faces = faces + vertex_offset
                vertex_offset += vertices.shape[0]
                
                all_vertices.append(vertices)
                all_faces.append(faces)
                if colors is not None:
                    all_colors.append(colors)
            
            # Free chunk memory
            del chunk_v_attrs_d, chunk_weights_d, sdf_d, deform_d, x_nx3, chunk_verts, chunk_cube
            gc.collect()
        
        if len(all_vertices) == 0:
            # Return empty mesh
            return MeshExtractResult(
                vertices=torch.zeros((0, 3), device=device),
                faces=torch.zeros((0, 3), dtype=torch.long, device=device),
                vertex_attrs=torch.zeros((0, 6), device=device) if self.use_color else None,
                res=self.res
            )
        
        # Merge all chunks
        vertices = torch.cat(all_vertices, dim=0)
        faces = torch.cat(all_faces, dim=0)
        colors = torch.cat(all_colors, dim=0) if len(all_colors) > 0 else None
        
        mesh = MeshExtractResult(
            vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res
        )
        return mesh
