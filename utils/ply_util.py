# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import open3d as o3d
import numpy as np


def save_ply(save_path, points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    if colors is not None:
        if isinstance(colors, (int, float)):
            rgb = np.array([[colors, colors, colors]] * len(points))
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    o3d.io.write_point_cloud(save_path, pcd)


def show_ply(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    if colors is not None:
        if isinstance(colors, (int, float)):
            rgb = np.array([[colors, colors, colors]] * len(points))
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    o3d.visualization.draw_geometries([pcd])


def load_ply(ply_file) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.array(pcd.points)
    return points
