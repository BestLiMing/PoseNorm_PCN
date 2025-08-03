# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import numpy as np
import open3d as o3d


def compute_normals(points, radius=0.1, max_nn=30):
    """Compute normals for a point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.asarray(pcd.normals)
    return normals


def back_points_mapping(tpose_front_scan_points: np.ndarray, tpose_back_scan_points: np.ndarray,
                        pose_front_scan_points: np.ndarray) -> np.ndarray:
    tpose_front = np.asarray(tpose_front_scan_points)
    tpose_back = np.asarray(tpose_back_scan_points)
    pose_front = np.asarray(pose_front_scan_points)

    normals_tpose = compute_normals(tpose_front)
    offset_tpose = tpose_back - tpose_front

    normals_posed = compute_normals(pose_front)
    centroid_tpose = np.mean(tpose_front, axis=0)
    centroid_posed = np.mean(pose_front, axis=0)

    H = (tpose_front - centroid_tpose).T @ (pose_front - centroid_posed)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_posed - R @ centroid_tpose

    offset_posed = (R @ offset_tpose.T).T
    pose_back_scan_points = pose_front + offset_posed
    return pose_back_scan_points
