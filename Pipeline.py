# ============================================================
#  Project    : PoseNorm_PCN
#  File       : Pipeline.py
#  Author     : Ming Li
#  Copyright  : (c) 2025 by Ming Li. All rights reserved.
#  Email      : helloming@shu.edu.cn
#  License    : For academic and research use only.
#  Description: Shanghai University
# ============================================================

import os
import sys
import time
import torch
import open3d as o3d
import numpy as np
import trimesh
import argparse
from shutil import copy
from typing import Literal
from os.path import exists, join, dirname, basename, splitext

from config.load_config import load_config
from config.LoggerConfig import setup_logger

config, project_root = load_config()
sys.path.append(project_root)
logger = setup_logger(name='pipeline', log_dir=join(project_root, 'logs'))

from eval import PoseCorr, BackGeo
from utils.sample_points import sample_points
from lib.extract_front_back_points_from_obj.extract_points import front_points_extraction
from lib.normal_conjugate_transfer.back_points_mapping import back_points_mapping
from utils.show_save_o3d import show_points, save_ply
from lib.points_mesh_standarization import sv_points, sv_mesh

'''load parameter'''
BASE_PARAM = config.get('base_param', {})
NUM_PARTS = int(BASE_PARAM.get('num_parts', 14))
NUM_POINTS = int(BASE_PARAM.get('num_points', 10000))
DEVICE = BASE_PARAM.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
OUT_PATH = join(project_root, BASE_PARAM.get('out_path', 'results'), 'pipeline')

PIPELINE_PARAM = config.get('pipeline', {})
POSECORR_PART_WEIGHTING = bool(PIPELINE_PARAM.get('PoseCorr_part_weighting', True))
BACKGEO_PART_WEIGHTING = bool(PIPELINE_PARAM.get('BackGeo_part_weighting', True))
EXP_SUFFIX = PIPELINE_PARAM.get('exp_suffix', None)

TGT_HEIGHT = 1.5
TGT_CENTER = 0.0


def get_input(file: str, scan_type: str = "single") -> dict:
    assert exists(file), "The scanned file does not exist."
    name, suffix = splitext(basename(file))
    if suffix == '.ply':
        pcd = o3d.io.read_point_cloud(file)
        vertices = np.array(pcd.points)
        scan_points = sample_points(vertices, NUM_POINTS)
    elif suffix == '.npz':
        data = np.load(file)
        scan_points = data['scan_points']
    elif suffix == '.obj':
        mesh = trimesh.load_mesh(file)
        vertices = np.array(mesh.vertices)
        faces = mesh.faces
        if scan_type == 'single':
            scan_points = front_points_extraction(file)
        else:
            mesh = trimesh.load_mesh(file)
            scan_points = mesh.sample(NUM_POINTS)
    else:
        raise "Illegal file type"

    scan_points_sv, scale, translation = sv_points(scan_points, tgt_height=TGT_HEIGHT, tgt_center=TGT_CENTER)
    mesh_sv = None
    if suffix == '.obj':
        vertices_sv = vertices * scale + translation
        mesh_sv = trimesh.Trimesh(vertices=vertices_sv, faces=faces)

    return {
        'scan_points': scan_points_sv,
        'file_name': name,
        'file_suffix': suffix,
        "scale": scale,
        "translation": translation,
        "sv_mesh": mesh_sv
    }


def pipeline(scan_file, gender: str = 'male', save: bool = True, save_path: str = None):
    """
    Three-dimensional implicit reconstruction of the human body based on Gaussian geometric diffusion with surface-dense correspondence relations
    :param scan_file: Scan point cloud files, .ply .obj .npz
    :param scan_type: Scan point cloud type, frontal scan or full body scan
    :param gender: Gender, default male
    :param save: Save optimized data and measurement data, default True
    :return:
    """
    logger.info(f'Start reconstruction pipeline')

    logger.info(f'Get input, gender: {gender}, file: {scan_file}')

    if isinstance(scan_file, np.ndarray):
        scan_points_np = scan_file
        file_name = ""
    else:
        inputs_dict = get_input(scan_file)
        scan_points_np = inputs_dict['scan_points']
        file_name = inputs_dict['file_name']

        scan_mesh_path = None
        sv_mesh_save_path = None
        if splitext(basename(scan_file))[1] == '.obj':
            scan_mesh_path = join(save_path, basename(scan_file))
            copy(scan_file, scan_mesh_path)
            mesh = trimesh.load_mesh(scan_mesh_path)
            sv_mesh = inputs_dict['sv_mesh']
            sv_mesh_save_path = join(dirname(scan_mesh_path), "sv_mesh.obj")
            sv_mesh.export(sv_mesh_save_path)

    print(
        "Start pipeline\n"
        f"base params: {BASE_PARAM}\n"
        f"pipeline params: {PIPELINE_PARAM}"
    )

    if save_path is not None:
        global OUT_PATH
        OUT_PATH = save_path
        os.makedirs(OUT_PATH, exist_ok=True)

    if save:
        save_ply(f"{OUT_PATH}/input_points.ply", points=scan_points_np)

    pipeline_start_time = time.time()

    '''PoseCorr-Net inference'''
    logger.info(f'Start PoseCorr, part weighting: {POSECORR_PART_WEIGHTING}')
    tpose_scan_points_np, _, tpose_scan_colors_np, PoseCorr_time_consumption = PoseCorr.PoseCorr_inference(
        scan_points=scan_points_np, num_parts=NUM_PARTS, part_weighting=POSECORR_PART_WEIGHTING, exp_suffix=EXP_SUFFIX,
        device=DEVICE, save=save, file_name=file_name, out_path=OUT_PATH)
    tpose_scan_full_points, pose_scan_full_points = tpose_scan_points_np, scan_points_np
    BackGeo_time_consumption = 0

    '''BackGeo-Net inference'''
    logger.info(f'Start BackGeo, part weighting: {BACKGEO_PART_WEIGHTING}')
    tpose_scan_back_points_np, _, _, BackGeo_time_consumption = BackGeo.BackGeo_inference(
        scan_points=tpose_scan_points_np, num_parts=NUM_PARTS, part_weighting=BACKGEO_PART_WEIGHTING,
        exp_suffix=EXP_SUFFIX, device=DEVICE, save=save, file_name=file_name, out_path=OUT_PATH)
    # back points mapping
    pose_scan_back_points = back_points_mapping(tpose_front_scan_points=tpose_scan_points_np,
                                                tpose_back_scan_points=tpose_scan_back_points_np,
                                                pose_front_scan_points=scan_points_np)
    tpose_scan_full_points = np.concatenate((tpose_scan_points_np, tpose_scan_back_points_np))
    pose_scan_full_points = np.concatenate((scan_points_np, pose_scan_back_points))
    pose_scan_full_colors = np.concatenate((tpose_scan_colors_np, tpose_scan_colors_np))

    pose_scan_back_points_save_path = join(OUT_PATH, f'{file_name}_back_points.ply')
    save_ply(pose_scan_back_points_save_path, pose_scan_back_points, tpose_scan_colors_np)
    pose_scan_full_save_path = join(OUT_PATH, f"{file_name}_completion.ply")
    save_ply(pose_scan_full_save_path, pose_scan_full_points, pose_scan_full_colors)
    pipeline_time = round(time.time() - pipeline_start_time, 2)
    logger.info(
        f"Pipeline execution completed.\n"
        f"PoseCorr-Net time: {PoseCorr_time_consumption}s,\n"
        f"BackGeo-Net time: {BackGeo_time_consumption}s,\n"
        f"pipeline time: {pipeline_time}s"
    )
    return pose_scan_full_points, pose_scan_full_colors, pipeline_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PoseNorm_PCN")
    parser.add_argument('--scan_file', '--f', type=str, required=True, help="scan file")
    parser.add_argument('--gender', '--g', type=str, choices=['male', 'female'], required=True, help="gender")
    parser.add_argument('--save_path', '--s', type=str, default=f"{dirname(__file__)}/results", help="save path")
    args = parser.parse_args()

    # args = parser.parse_args([
    #     '--scan_file', join(dirname(__file__), "test_data", "00032_longshort_ATUsquat.000001.npz"),
    #     '--gender', 'male',
    #     '--save_path', f"{dirname(__file__)}/results"
    # ])

    pipeline(scan_file=args.scan_file, gender=args.gender, save=True, save_path=args.save_path)
