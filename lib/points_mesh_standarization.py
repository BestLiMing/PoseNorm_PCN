# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import numpy as np
import torch
import trimesh
from copy import deepcopy


def sv_points(points: np.ndarray, tgt_height: float = 1.5, tgt_center: float = 0.0):
    '''points standardization'''
    height = np.max(points[:, 1]) - np.min(points[:, 1])
    scale = tgt_height / height
    points_scaled = points * scale
    center = (np.max(points_scaled, axis=0) + np.min(points_scaled, axis=0)) / 2
    translation = tgt_center - center
    points_centered = points_scaled + translation
    return points_centered, scale, translation


def sv_points_tensor(points: torch.Tensor, tgt_height: float = 1.5, tgt_center: float = 0.0):
    '''points standardization'''
    points_np = points.detach().cpu().numpy()
    points_np_centered, scale, translation = sv_points(points_np, tgt_height, tgt_center)
    points_th = torch.from_numpy(points_np_centered)
    return points_th, scale, translation


def sv_mesh(mesh: trimesh.Trimesh, tgt_height: float = 1.5, tgt_center: float = 0.0):
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    vertices_centered, scale, translation = sv_points(vertices, tgt_height, tgt_center)
    normalized_mesh = trimesh.Trimesh(
        vertices=vertices_centered,
        faces=faces,
        process=False
    )
    if mesh.visual is not None:
        normalized_mesh.visual = deepcopy(mesh.visual)
    return normalized_mesh, scale, translation


if __name__ == "__main__":
    from os.path import dirname

    # mesh_file = r"H:\W-Paper-1in2-2025-GaussianDiffusionImplicitReconstruction\GDIR\paper_exp\pipeline_exp\results\pipeline_exp2\full_sv\body_points_0311_results\optimization\cal_50_hot_100\mesh_smplx_opt_pose_smpl_mmw.obj"
    # mesh = trimesh.load_mesh(mesh_file)
    # mesh_sv, scale, translation = sv_mesh(mesh)
    # mesh_sv.export(f"{dirname(mesh_file)}/mesh_smplx_opt_pose_smpl_mmw_sv.obj")
    #
    # mesh_gt_file = r"H:\W-Paper-1in2-2025-GaussianDiffusionImplicitReconstruction\GDIR\paper_exp\pipeline_exp\selected_data\0311\mesh_smplx.obj"
    # mesh_gt = trimesh.load_mesh(mesh_gt_file)
    # mesh_gt_sv, scale_gt, translation_gt = sv_mesh(mesh_gt)
    # mesh_gt_sv.export(f"{dirname(mesh_gt_file)}/mesh_smplx_sv.obj")

    scan_file = r"H:\W-Paper-1in2-2025-GaussianDiffusionImplicitReconstruction\GDIR\paper_exp\pipeline_exp\selected_data\0311\0311.obj"
    scan_mesh = trimesh.load_mesh(scan_file)
    scan_mesh_sv, scan_scale, scan_translation = sv_mesh(scan_mesh)
    scan_mesh_sv.export(f"{dirname(scan_file)}/scan_mesh_sv.obj")
