# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import os
import cv2
from pathlib import Path
from os.path import basename, dirname, join, splitext, exists, split
import torch
from typing import List, Tuple, Dict, Optional
import numpy as np
import trimesh
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
)

import open3d as o3d
import argparse
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import shutil

FRONT_CAMERA_PARAMS = {
    'dist': 2.0,
    'elev': 0,
    'azim': 0,
    'at': ((0, 0, 0),),
    'up': ((0, 1, 0),)
}
BACK_CAMERA_PARAMS = {
    'dist': -2.0,
    'elev': 0,
    'azim': 0,
    'at': ((0, 0, 0),),
    'up': ((0, 1, 0),)
}
size = 512
num_points = 10000


class GPUMemoryManager:
    @staticmethod
    def empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DepthProcessor:
    def __init__(self, device: torch.device = None, front: bool = True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = 512
        if front:
            self.camera_params = FRONT_CAMERA_PARAMS
        else:
            self.camera_params = BACK_CAMERA_PARAMS

    def _load_mesh(self, model_path: Path) -> Meshes:
        try:
            if model_path.suffix == '.obj':
                verts, faces = load_obj(model_path, device=self.device)[:2]
                faces = faces.verts_idx
            elif model_path.suffix == '.ply':
                verts, faces = load_ply(model_path)
            else:
                raise ValueError(f"Unsupported format: {model_path.suffix}")

            return Meshes(
                verts=[verts.to(self.device)],
                faces=[faces.to(torch.int64).to(self.device)]
            )
        except Exception as e:
            print(f"Model loading failed {model_path}: {str(e)}")
            raise

    def _render_depth(self, mesh: Meshes, size: int = 512) -> Tuple[torch.Tensor, OpenGLPerspectiveCameras]:
        try:
            R, T = look_at_view_transform(**self.camera_params, device=self.device)
            cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=T)
            raster_settings = RasterizationSettings(image_size=size, faces_per_pixel=1, perspective_correct=False)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(mesh)
            depth_map = fragments.zbuf.squeeze()
            return depth_map, cameras
        except Exception as e:
            print(f"Failed to render depth map: {str(e)}")
            raise

    def _depth_to_point_cloud(self, depth_map: torch.Tensor, cameras: OpenGLPerspectiveCameras, num_points: int,
                              sampling_mode: str) -> np.ndarray:
        try:
            height, width = depth_map.shape
            u, v = torch.meshgrid(torch.arange(width, device=self.device), torch.arange(height, device=self.device),
                                  indexing='xy')
            uv = torch.stack([u.flatten(), v.flatten()], dim=1)
            depths = depth_map[uv[:, 1], uv[:, 0]]
            valid_mask = torch.isfinite(depths) & (depths > 0)
            uv_valid = uv[valid_mask]
            depths_valid = depths[valid_mask]

            if uv_valid.shape[0] == 0:
                raise ValueError("No valid data points in depth map")

            if num_points is not None:
                if sampling_mode == 'random':
                    indices = torch.randperm(uv_valid.shape[0], device=self.device)[:num_points]
                elif sampling_mode == 'uniform':
                    step = max(1, uv_valid.shape[0] // num_points)
                    indices = torch.arange(0, uv_valid.shape[0], step, device=self.device)[:num_points]
                else:
                    raise ValueError(f"Unknown sampling mode: {sampling_mode}")
                uv_sampled = uv_valid[indices]
                depths_sampled = depths_valid[indices]
            else:
                uv_sampled = uv_valid
                depths_sampled = depths_valid

            proj_matrix = cameras.get_projection_transform().get_matrix()[0]
            fx = proj_matrix[0, 0] * (width / 2)
            fy = proj_matrix[1, 1] * (height / 2)
            cx = (proj_matrix[0, 2] + 1) * (width / 2)
            cy = (proj_matrix[1, 2] + 1) * (height / 2)

            x = (uv_sampled[:, 0] - cx) * depths_sampled / fx
            y = (uv_sampled[:, 1] - cy) * depths_sampled / fy
            z = depths_sampled

            R = cameras.R[0]
            T = cameras.T[0]
            R_fix = torch.tensor([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ], dtype=torch.float32, device=self.device)
            R_corrected = torch.mm(R, R_fix)
            points_camera = torch.stack([x, y, z], dim=1)
            points_world = torch.mm(points_camera, R_corrected) + T

            return points_world.cpu().numpy()
        except Exception as e:
            print(f"Point cloud generation failed: {str(e)}")
            raise

    def _get_camera_intrinsics(self, cameras: OpenGLPerspectiveCameras, size: int = 512):
        proj_matrix = cameras.get_projection_transform().get_matrix()[0]
        fx = proj_matrix[0, 0] * size / 2
        fy = proj_matrix[1, 1] * size / 2
        cx = size / 2 - proj_matrix[0, 2] * size / 2
        cy = size / 2 - proj_matrix[1, 2] * size / 2

        return torch.tensor([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], device=self.device)

    def process_single(self, input_path, num_points: int, sampling_mode: str, front_mode: str = 'scan',
                       save: bool = True, save_path: str = None, exp: bool = False):
        try:
            if isinstance(input_path, trimesh.Trimesh):
                mesh = Meshes(verts=torch.tensor(input_path.vertices, dtype=torch.float32).unsqueeze(0).to(self.device),
                              faces=torch.tensor(input_path.faces, dtype=torch.int64).unsqueeze(0).to(self.device))
                source_name = "trimesh_input"
            else:
                if not exists(input_path):
                    raise FileNotFoundError(f"The input path does not exist.: {input_path}")
                input_path = Path(input_path)
                if input_path.suffix.lower() in ['.obj', '.ply']:
                    mesh = self._load_mesh(input_path)
                    source_name = input_path.stem
                else:
                    source_name = input_path.name
                    mode = 'scaled' if front_mode == 'scan' else 'scaled_naked'
                    input_file = input_path / "scaled" / f"{source_name}_{mode}.obj"
                    if "cape_data" not in input_path.parts:
                        raise ValueError(f"Key identifiers missing from directory structure: cape_data")

                    mesh = self._load_mesh(input_file)
            depth_map, cameras = self._render_depth(mesh, size=self.size)
            pointcloud = self._depth_to_point_cloud(depth_map, cameras, num_points, sampling_mode)

            if save:
                if save_path is None:
                    save_path = input_path
                output_path = join(save_path, "front_views")
                os.makedirs(output_path, exist_ok=True)
                output_file = join(output_path, "front.ply")
                output_map = join(output_path, "front.png")
                if exists(output_file) and exp is False:
                    print(f"Skip processed files: {output_file}")
                    return output_file
                self._save_points(pointcloud, output_file, show=False)
                # self._svae_depth_map(depth_map, output_map)
                return output_file
            else:
                return pointcloud
        except Exception as e:
            print(f"Processing failed {input_path}: {str(e)}")
        return None

    def _save_points(self, points, save_path, camera_pos=None, show: bool = False):
        pcd = o3d.geometry.PointCloud()
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError("Point cloud data should be a two-dimensional array with shape (n, 3).")

        pcd.points = o3d.utility.Vector3dVector(points)
        geometries = [pcd]
        if show:
            if camera_pos is not None:
                eye = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                eye.paint_uniform_color([1.0, 0.0, 0.0])
                eye.translate(camera_pos)
                geometries.append(eye)
            o3d.visualization.draw_geometries(geometries)

        try:
            o3d.io.write_point_cloud(save_path, pcd)
        except Exception as e:
            print(f"Error saving point cloud: {e}")

    def _save_depth_map(self, depth_map, save_path):
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        if depth_map.dtype != np.uint8:
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_map_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            print(f"Original Min: {depth_min}, Original Max: {depth_max}")
            print(f"Normalized Min: {depth_map_normalized.min()}, Normalized Max: {depth_map_normalized.max()}")
        else:
            depth_map_normalized = depth_map
        cv2.imwrite(save_path, depth_map_normalized)

    def camera_position_calculate(self, R, T):
        R_matrix = R[0]
        T_vector = T[0]
        camera_pos = -torch.mm(R_matrix.T, T_vector.unsqueeze(1)).squeeze()
        camera_pos_np = camera_pos.cpu().numpy()
        return camera_pos_np


def delete_front_files(input: Path):
    output_path = join(input, "front_views")
    if exists(output_path):
        try:
            shutil.rmtree(output_path)
            print(f"Delete folder {output_path} success")
            return True
        except Exception as e:
            print(f"Delete folder {output_path} fail")
            return False
    else:
        print(f"Folder {output_path} does not exist")
        return False


def worker_init():
    import signal
    try:
        sig = signal.SIGINT
        handler = signal.SIG_IGN
    except AttributeError:
        sig = signal.CTRL_C_EVENT
        handler = signal.SIG_DFL

    signal.signal(sig, handler)
    torch.cuda.empty_cache()
    import random
    random.seed()


def batch_processor(dataset_info: Path, num_points: int = 10000, sampling_mode: str = 'random',
                    workers: int = 6) -> Dict[str, List[Path]]:
    try:
        data = np.load(dataset_info, allow_pickle=True)
        base_path = Path(str(data["data_path"]))
        all_folders = [base_path / f for f in data["all_folders"]]
        processor = DepthProcessor()
        task_args = [
            (folder, num_points, sampling_mode)
            for folder in all_folders
        ]
        success = []
        failure = []
        with mp.Pool(
                processes=workers,
                initializer=worker_init
        ) as pool:
            try:
                results = []
                wrapped_func = partial(_process_wrapper, processor=processor)
                chunksize = max(1, len(task_args) // (workers * 10))

                with tqdm(total=len(task_args), desc="processing progress") as pbar:
                    for result in pool.imap_unordered(wrapped_func, task_args, chunksize=chunksize):
                        if result is not None:
                            success.append(result)
                            pbar.update()
                            pbar.set_postfix_str(f"success rate: {len(success) * 100 / (len(success) + len(failure)):.1f}%")
                        else:
                            failure.append(result)
                            pbar.update()
            except KeyboardInterrupt:
                print("User interruption, terminating work process...")
                pool.terminate()
                pool.join()
                raise

        result_file = join(dirname(__file__), "../../preprocess/CAPE/CAPE_front_files.npz")
        np.savez(
            str(result_file),
            success=np.array([str(p) for p in success], dtype=object),
            failure=np.array([str(p[0]) for p in failure], dtype=object)
        )

        return {
            "success": success,
            "failure": failure,
            "result_file": result_file
        }
    except Exception as e:
        print(f"Batch processing failed: {str(e)}")
        raise


def _process_wrapper(args: Tuple[Path, int, str], processor: DepthProcessor) -> Optional[Path]:
    input_path, num_points, sampling_mode = args
    try:
        return processor.process_single(input_path, num_points, sampling_mode)
    except Exception as e:
        print(f"Processing failed {input_path}: {str(e)}")
        return None
    finally:
        GPUMemoryManager.empty_cache()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="extract front point cloud")
    parser.add_argument('--CAPE_all_folders', type=str,
                        default=r"H:\W-Paper-1in3-2025-IHR\Net\dataset_preprocess\CAPE\CAPE_all_folders.npz",
                        help="Folder list file path")
    parser.add_argument('--num_points', default=None, help="Number of extraction points")
    parser.add_argument('--sampling_mode', type=str, default='random', choices=["random", "uniform"],
                        help="sampling mode")
    parser.add_argument("--workers", type=int, default=6, help="Number works")
    args = parser.parse_args()

    # if not exists(args.CAPE_all_folders):
    #     raise FileNotFoundError(f"数据集信息文件不存在: {args.CAPE_all_folders}")

    # 执行处理
    # try:
    #     results = batch_processor(
    #         dataset_info=args.CAPE_all_folders,
    #         num_points=args.num_points,
    #         sampling_mode=args.sampling_mode,
    #         workers=args.workers
    #     )
    #     print(f"处理完成，结果保存至: {results['result_file']}")
    #     print(f"成功: {len(results['success'])}")
    #     print(f"失败: {len(results['failure'])}")
    # except Exception as e:
    #     print(f"处理发生严重错误: {str(e)}")
    #     exit(1)

    processor = DepthProcessor().process_single(
        # r"H:\W-Paper-1in3-2025-IHR\cape_data\00032\longshort_ATUsquat\longshort_ATUsquat.000001",
        # r"H:\W-Paper-1in3-2025-IHR\cape_data\00032\longshort_pose_model\longshort_pose_model.000025",
        r"H:\Thesis\figures\figure4.1\longshort_ATUsquat.000001",
        num_points=args.num_points, sampling_mode=args.sampling_mode, save=True)

    # 多进程删除正面点云文件
    # worker_func = partial(
    #     delete_front_files,
    # )
    # with mp.Pool(processes=6) as pool:
    #     rs = pool.map(worker_func, all_folder_list)
    # success_count = sum(1 for status in rs if status)
    # print(f"Processed {folder_count} folders. Success: {success_count}, Failed: {folder_count - success_count}")
