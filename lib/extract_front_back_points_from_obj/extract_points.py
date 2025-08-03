# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import open3d as o3d
import numpy as np
from os.path import dirname
import sys


project_path = dirname(dirname(dirname(__file__)))
sys.path.append(project_path)

ib_checkpoints = r"H:\W-Paper-1in3-2025-InBody\Net\Phase2_InBodyNet\experiments\InBody_3060\checkpoints\best_checkpoint.pt"


def front_and_back_points_extraction(obj_file):
    from lib.extract_front_back_points_from_obj.get_front_point_from_obj import DepthProcessor
    front_points = DepthProcessor(front=True).process_single(obj_file, 20000, 'random', save=False, exp=True)
    back_points = DepthProcessor(front=False).process_single(obj_file, 20000, 'random', save=False, exp=True)
    back_points[:, 2] -= 3
    full_points = np.concatenate((np.array(front_points), np.array(back_points)))
    return full_points


def front_points_extraction(obj_file):
    from lib.extract_front_back_points_from_obj.get_front_point_from_obj import DepthProcessor
    front_points = DepthProcessor(front=True).process_single(obj_file, 10000, 'random', save=False, exp=True)
    return np.array(front_points)


def save_ply(points, save_path):
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([full_pcd])
    o3d.io.write_point_cloud(save_path, full_pcd)


if __name__ == "__main__":
    # cases = r"H:\W-Paper-1in3-2025-InBody\Net\paper\experiments\E_real_scan_experiment\real_cases"
    # obj = r"BUFF_00005_shortlong_hips_000008"
    # obj = r"BUFF_03223_shortlong_hips_000001"
    # obj = r"Multi-Garment_125611500511320"
    # obj = r"Multi-Garment_125611502992247"
    # obj = r"THuman2.0_0181"
    # obj = r"THuman2.0_0455"

    # file = rf"{cases}\{obj}\InBody_results\_body_mesh.ply"
    # obj_file = file.replace(".ply", ".obj")
    # ply_to_obj_with_trimesh(file, obj_file)

    # case = r"H:\W-Paper-1in3-2025-InBody\Net\paper\experiments\D_reconstruction_evaluation_experiment\cases_CAPE\coarse"
    # obj = r"00032_longshort_hips.000120"
    # for obj in os.listdir(case):
    #     print(f"start: {obj}")
    # scan_obj_file = rf"{case}\{obj}\scaled.obj"
    # full_points_file = rf"{case}\{obj}\full_points.ply"

    # scan_obj_file = rf"{case}\{obj}\body_mesh.obj"
    # full_points_file = rf"{case}\{obj}\body_points.ply"

    # scan_obj_file = rf"{case}\{obj}\{obj.split('_', 1)[1]}_scaled_naked.obj"
    # full_points_file = rf"{case}\{obj}\body_points_naked.ply"
    # if not exists(scan_obj_file):
    #     print(f"exist: {scan_obj_file}")
    #     continue
    # scan_full_points = front_and_back_points_extraction(scan_obj_file)
    # save_ply(scan_full_points, full_points_file)

    # body_mesh_path = rf"{case}\{obj}\body_mesh.obj"
    # body_points = front_and_back_points_extraction(body_mesh_path)
    # save_ply(body_points, rf"{case}\{obj}\body_points.ply")

    # ply_to_obj_with_trimesh(body_mesh,body_mesh_obj)

    # single
    # obj = "00096_jerseyshort_pose_model.000045"
    # obj = "00032_longshort_soccer.000065"
    # obj = "00096_shirtlong_tilt_twist_left.000208"
    # scan_obj_file = rf"{case}\{obj}\scaled.obj"
    # full_points_file = rf"{case}\{obj}\full_points.ply"

    # scan_obj_file = r"J:\data\inbody\00122-shortlong_ATUsquat\scaled\shortlong_ATUsquat.000186_scaled.obj"
    # full_points_file = rf"{dirname(scan_obj_file)}\full_points.ply"
    # scan_full_points = front_and_back_points_extraction(scan_obj_file)
    # save_ply(scan_full_points, full_points_file)
    #
    # body_obj_file = r"J:\data\inbody\00122-shortlong_ATUsquat\scaled\shortlong_ATUsquat.000186_scaled_naked.obj"
    # body_points_file = rf"{dirname(body_obj_file)}\body_points.ply"
    # body_full_points = front_and_back_points_extraction(body_obj_file)
    # save_ply(body_full_points, body_points_file)

    # body_mesh_obj = r"J:\data\inbody\00032-shortshort_punching\scaled\shortshort_punching.000078_scaled_naked.obj"
    # body_points = front_and_back_points_extraction(body_mesh_obj)
    # save_ply(body_points, fr"{dirname(body_mesh_obj)}\body_points.ply")

    scan_obj_file = r"H:\W-Paper-1in2-2025-GaussianDiffusionImplicitReconstruction\GDIR\paper_exp\pipeline_exp\selected_data\0210\0210.ply"
    front_points = front_points_extraction(scan_obj_file)
    front_points_file = rf"{dirname(scan_obj_file)}\front_points.ply"
    save_ply(front_points, front_points_file)

    # mesh = trimesh.load(scan_obj_file)
    # points = mesh.sample(10000)
    # show_points(points)
    # save_ply(points, f"{dirname(scan_obj_file)}/full.ply")
