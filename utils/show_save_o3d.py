# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import open3d as o3d
import numpy as np


def create_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd


def show_points(points, colors=None):
    pcd = create_pcd(points, colors)
    o3d.visualization.draw_geometries([pcd])


def save_ply(save_path, points, colors=None):
    pcd = create_pcd(points, colors)
    o3d.io.write_point_cloud(save_path, pcd)
