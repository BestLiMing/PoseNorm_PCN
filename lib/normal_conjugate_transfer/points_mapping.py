# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import numpy as np
import trimesh
import os
import sys
import pickle as pkl
from os.path import join, exists, dirname, basename, splitext
from psbody.mesh import Mesh
from sklearn.neighbors import KDTree

ROOT = dirname(dirname(dirname(__file__)))
sys.path.append(ROOT)
from utils.show_save_o3d import show_points, create_pcd, save_ply
from lib.generate_colors import generate_colors

SMPL_PARTS_DENSE = join(ROOT, 'assets', 'smpl_parts_dense.pkl')
assert exists(SMPL_PARTS_DENSE)

BASE_COLORS = generate_colors(14)


def get_smpl_part(mesh: trimesh.Trimesh, points: np.ndarray) -> [np.ndarray, np.ndarray]:
    with open(SMPL_PARTS_DENSE, 'rb') as f:
        dat = pkl.load(f, encoding='latin-1')
    smpl_parts = np.zeros((6890, 1))
    for n, k in enumerate(dat):
        smpl_parts[dat[k]] = n
    smpl_mesh = Mesh(v=mesh.vertices, f=mesh.faces)
    vertex_tree = KDTree(smpl_mesh.v)
    _, vertex_indices = vertex_tree.query(points)
    indices = vertex_indices.ravel()
    part_labels = smpl_parts[indices].flatten().astype(np.int32)
    color_indices = part_labels % len(BASE_COLORS)
    color_indices = np.asarray(color_indices, dtype=np.int32)
    colors = BASE_COLORS[color_indices]
    return part_labels, colors


def point_mapping(tpose_body_mesh: trimesh.Trimesh, tpose_scan_points: np.ndarray, pose_scan_points: np.ndarray,
                  save_path: str = None, file_name: str = "") -> dict:
    # processing body points
    scan_tpose_points_to_mesh, _, _ = tpose_body_mesh.nearest.on_surface(tpose_scan_points)
    part_labels, colors = get_smpl_part(tpose_body_mesh, scan_tpose_points_to_mesh)

    offsets = scan_tpose_points_to_mesh - tpose_scan_points
    pose_body_points = pose_scan_points + offsets

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        # T-pose mesh
        tpose_body_mesh_save_path = join(save_path,
                                         f"{file_name}_tpose_body_mesh.obj" if file_name.strip() else "tpose_body_mesh.obj")
        tpose_body_mesh.export(tpose_body_mesh_save_path)

        # The corresponding points of scan t pose points near the mesh
        tpose_scan_near_mesh_points_save_path = join(save_path,
                                                     f"{file_name}_tpose_scan_near_mesh_points.ply" if file_name.strip() else "tpose_scan_near_mesh_points.ply")
        save_ply(tpose_scan_near_mesh_points_save_path, scan_tpose_points_to_mesh, colors)

        # Pose body points
        pose_body_points_save_path = join(save_path,
                                          f"{file_name}_pose_body_points.ply" if file_name.strip() else "pose_body_points.ply")
        save_ply(pose_body_points_save_path, pose_body_points, colors)

        # parts
        pose_body_parts_save_path = join(save_path,
                                         f"{file_name}_pose_body_parts.npz" if file_name.strip() else "pose_body_parts.npz")
        np.savez(pose_body_parts_save_path, part=part_labels, colors=colors)

    return {
        'pose_body_points': pose_body_points,
        'parts': part_labels,
        'colors': colors
    }


if __name__ == "__main__":
    pose_scan = r"H:\W-Paper-1in3-2025-IHR\diffusion_implicit_reconstruction\data\tpose_corr_full\00032_longshort_ATUsquat.000001.npz"
    body_scan = r"H:\W-Paper-1in3-2025-IHR\diffusion_implicit_reconstruction\data\body_occs_part_5000\00032_longshort_ATUsquat.000001.npz"
    pose_data = np.load(pose_scan)
    body_data = np.load(body_scan)

    scan_points = pose_data['scan_points'].astype(np.float64)
    scan_corr = pose_data['correspondences'].astype(np.float64)
    gender = body_data['gender'].astype(str)
    body_points = body_data['body_points'].astype(np.float64)
    # print(type(scan_points), type(scan_corr), type(gender), type(body_points))
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(scan_corr)
    # o3d.visualization.draw_geometries([pcd])

    save_path = r"H:\W-Paper-1in3-2025-IHR\diffusion_implicit_reconstruction\results"
    name = splitext(basename(pose_scan))[0]
    point_mapping(body_points=body_points, gender=gender, corr_points=scan_corr, scan_points=scan_points,
                  save_path=save_path, file_name=name)

    # scan_pcd = create_pcd(barycentric)
    # scan_pcd.paint_uniform_color((1, 0, 0))
    # body_pcd = create_pcd(scan_points)
    # body_pcd.paint_uniform_color((0, 1, 0))
    # o3d.visualization.draw_geometries([scan_pcd, body_pcd])
