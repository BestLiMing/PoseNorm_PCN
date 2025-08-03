# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import os
import sys
import torch
import numpy as np
import time
from os.path import exists, join, dirname, basename, splitext
from models.Builder import Registers
from config.load_config import load_config

config, project_root = load_config()

sys.path.append(project_root)
from utils.load_checkpoints import load_checkpoints
from utils.ply_util import show_ply, load_ply, save_ply
from utils.SMPL_parts_dense import smpl_parts_colors
from lib.extract_front_back_points_from_obj.get_front_point_from_obj import DepthProcessor


def save_results(input_np: np.ndarray, logics_np: np.ndarray, colors_np: np.ndarray, parts_np: np.ndarray,
                 data_type: str, file_name: str, weighting_suffix: str, out_path: str = join(project_root, 'results')):
    save_path = join(out_path, f'PoseCorr_{data_type}')
    os.makedirs(save_path, exist_ok=True)
    pose_corr_save_path = join(save_path, f'{file_name}_PoseCorrNet_{data_type}_{weighting_suffix}.ply')
    save_ply(save_path=pose_corr_save_path, points=logics_np, colors=colors_np)
    pose_parts_save_path = join(save_path, f'{file_name}_PoseCorrNet_parts_{data_type}_{weighting_suffix}.npz')
    np.savez(file=pose_parts_save_path, parts=parts_np)
    input_save_path = join(save_path, f'{file_name}_input_{data_type}_{weighting_suffix}.ply')
    save_ply(input_save_path, input_np)
    print(f"The file is saved in: {save_path}")


def PoseCorr_inference(scan_points: np.ndarray, data_type: str = 'single', num_parts: int = 14,
                       part_weighting: bool = True, exp_suffix: str = None,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu', save: bool = True,
                       file_name: str = "", out_path: str = join(project_root, 'results')) -> [np.ndarray, np.ndarray,
                                                                                               np.ndarray, float]:
    register = Registers(model_type='PoseCorr', data_type=data_type, num_parts=num_parts)
    exp_name = register.exp_name_builder(exp_suffix=exp_suffix, part_weighting=part_weighting)
    checkpoint_path = join(project_root, 'experiments', exp_name, 'checkpoints', 'best_checkpoint.pt')
    assert exists(checkpoint_path), f"Checkpoint file does not exist: {checkpoint_path}"
    model = register.model_builder(part_weighting=part_weighting)
    model = load_checkpoints(model=model, checkpoint_path=checkpoint_path, device=device)
    points_th = torch.from_numpy(scan_points).unsqueeze(0).float().to(device)

    # summary_model(model, points_th)

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        out = model(points_th)
    time_consumption = round(time.time() - start_time, 2)
    # Analyzing the results of inference
    logics = out['logics'].permute(0, 2, 1).squeeze(0)
    parts = out['parts'].permute(0, 2, 1).squeeze(0)
    logics_np = logics.cpu().numpy()  # (N, 3)
    parts_np = torch.argmax(parts, dim=1).cpu().numpy()  # (N, )
    colors_np = smpl_parts_colors(parts_np=parts_np)
    if save:
        weighting_suffix = 'weighting' if part_weighting else 'no_weighting'
        save_results(input_np=scan_points, logics_np=logics_np, colors_np=colors_np, parts_np=parts_np,
                     data_type=data_type, file_name=file_name, weighting_suffix=weighting_suffix, out_path=out_path)
    print(
        f'The scanned pose has been successfully mapped to the standard space. Time consumed for inference: {time_consumption}s.')
    return logics_np, parts_np, colors_np, time_consumption


def posecorr_net(file: str):
    '''Analyze parameters'''
    base_param = config.get('base_param', {})
    num_parts = int(base_param.get('num_parts', 14))
    num_points = int(base_param.get('num_points', 10000))
    device = base_param.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    out_path = join(project_root, base_param.get('out_path', 'results'))
    model_type_eval_param = config.get('PoseCorr', {}).get('eval', {})
    part_weighting = bool(model_type_eval_param.get('part_weighting', True))
    data_type = model_type_eval_param.get('data_type', 'single')
    exp_suffix = '' if model_type_eval_param.get('exp_suffix', None) is None else model_type_eval_param.get(
        'exp_suffix', None)

    print(
        f'Construction of dense correspondence between scanned poses, scan type: single, weighting: {part_weighting}\n'
        f'base parameters: {base_param}\n'
        f'eval parameters: {model_type_eval_param}\n')

    # get input
    assert exists(file)
    file_name, suffix = splitext(basename(file))
    if suffix == '.npz':
        data = np.load(file)
        scan_points = data['scan_points']  # pose scan data
        correspondences = data['correspondences']  # tpose scan data
        parts = data['parts']  # parts label
    elif suffix == '.ply':
        scan_points = load_ply(file)
    elif suffix == '.obj':
        scan_points = DepthProcessor(front=True).process_single(file, num_points, 'random', save=False, exp=True)
    else:
        print('File type is not valid.')
        return

    print("PoseCorr-Net start inferring")
    PoseCorr_inference(scan_points=scan_points, data_type=data_type, num_parts=num_parts, part_weighting=part_weighting,
                       exp_suffix=exp_suffix, device=device, save=True, file_name=file_name, out_path=out_path)


if __name__ == "__main__":
    single_file = r"H:\W-Paper-1in2-2025-GaussianDiffusionImplicitReconstruction\GDIR\data\tpose_corr_single\00032_longshort_ATUsquat.000001.npz"

    posecorr_net(file=single_file)

    # data = np.load(file)
    # print(data)
    # scan_points = data['scan_points']
    # correspondences = data['correspondences']
    # show_ply(scan_points)
    # show_ply(correspondences)
