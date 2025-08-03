import numpy as np
import open3d as o3d
from os.path import join, exists, dirname, basename
import torch
from torch.utils.data import Dataset, DataLoader


class BaseDataLoader(Dataset):
    @staticmethod
    def check_valid_files(split_file, data_mode, data_root):
        split_data = np.load(split_file)
        root = split_data['root'] if data_root is None else data_root
        data = split_data[data_mode]
        valid_data = []
        for d in data:
            path = join(root, d)
            if exists(path):
                valid_data.append(path)
        assert len(valid_data) > 0, "Missing valid data"
        return valid_data

    def _apply_augmentation(self, points: torch.Tensor, transform) -> np.ndarray:
        '''Apply model enhancement and return numpy array'''
        device = points.device
        homog = torch.cat([points, torch.ones(len(points), 1, device=device)], dim=1)
        transformed = (transform @ homog.T).T[:, :3]
        return transformed.detach().cpu().numpy()

    @staticmethod
    def _generate_transform_matrix(aug_config) -> torch.Tensor:
        '''Generate global transformation matrix'''
        transform = torch.eye(4)
        # Rotation
        if aug_config['rotate']:
            max_angle_rad = aug_config['max_angle'] * (torch.pi / 180)
            angle = torch.zeros(3)

            for i, axis in enumerate(['x', 'y', 'z']):
                if axis in aug_config['rotate_axis']:
                    angle[i] = (torch.rand(1) * 2 - 1) * max_angle_rad

            Rx = torch.tensor([
                [1, 0, 0, 0],
                [0, torch.cos(angle[0]), -torch.sin(angle[0]), 0],
                [0, torch.sin(angle[0]), torch.cos(angle[0]), 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
            Ry = torch.tensor([
                [torch.cos(angle[1]), 0, torch.sin(angle[1]), 0],
                [0, 1, 0, 0],
                [-torch.sin(angle[1]), 0, torch.cos(angle[1]), 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
            Rz = torch.tensor([
                [torch.cos(angle[2]), -torch.sin(angle[2]), 0, 0],
                [torch.sin(angle[2]), torch.cos(angle[2]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
            transform = Rz @ Ry @ Rx @ transform

        # Translation
        if aug_config['translate']:
            translate = (torch.rand(3) * 2 - 1) * aug_config['max_translate']
            transform[:3, 3] += translate

        # Scaling
        if aug_config['scale']:
            scale = torch.rand(3) * 0.2 + 0.9
            scale_matrix = torch.eye(4)
            scale_matrix[:3, :3] = torch.diag(scale)
            transform = scale_matrix @ transform

        return transform


class CombineLoader(BaseDataLoader):
    def __init__(self, data_mode, model_type, split_file, num_sampling: int = None, augment: bool = False,
                 data_root: str = None):
        self.data_mode = data_mode
        self.model_type = model_type
        assert exists(split_file), "The split file does not exist"
        self.split_file = split_file
        self.num_sampling = num_sampling
        self.augment = augment

        self.current_data_list = self.check_valid_files(split_file=split_file, data_mode=data_mode, data_root=data_root)

        self.aug_config = {
            'rotate': True,
            'translate': True,
            'scale': False,
            'rotate_axis': ['x', 'y', 'z'],
            'max_angle': 30,
            'max_translate': 0.1,
        }

    def __len__(self):
        return len(self.current_data_list)

    def __getitem__(self, idx):
        file = self.current_data_list[idx]
        data = np.load(file)

        if self.model_type == 'PoseCorr':
            scan_points = data['scan_points'].astype(np.float32)
            correspondences = data['correspondences'].astype(np.float32)
            parts = data['parts'].astype(np.int64)
            if self.augment:
                scan_points_aug = torch.from_numpy(scan_points).float()
                transform = self._generate_transform_matrix(aug_config=self.aug_config)
                augmented_scan_points_aug = self._apply_augmentation(scan_points_aug, transform)
                scan_points = augmented_scan_points_aug.astype(np.float32)
            return {
                "scan_points": scan_points,
                "correspondences": correspondences,
                "parts": parts
            }
        elif self.model_type == 'BackGeo':
            correspondences = data['correspondences'].astype(np.float32)
            correspondences_back = data['correspondences_back'].astype(np.float32)
            parts = data['parts'].astype(np.int64)
            return {
                "correspondences": correspondences,
                "correspondences_back": correspondences_back,
                "parts": parts
            }
        else:
            raise 'The type of the model is not legal'


if __name__ == "__main__":
    # split_file = r'H:\W-Paper-1in3-2025-IHR\diffusion_implicit_reconstruction\data\tpose_corr\split_data.npz'
    # train_dataloader = TPoseDataLoader('train', split_file, 10000, augment=True)

    file = r"H:\W-Paper-1in2-2025-GaussianDiffusionImplicitReconstruction\GDIR\data\gaussian_diffusion_body_occs3_40000\00032_longshort_ATUsquat.000001.npz"
    data = np.load(file)

    points = np.array(data['points'])
    occs = np.array(data['occs'])
    parts = np.array(data['parts'])
    body_occs = np.array(data['body_occs'])
    print(points.shape, occs.shape, parts.shape, body_occs.shape)

    outer_points = points[body_occs == 0]
    middle_points = points[body_occs == 1]
    inner_points = points[body_occs == 2]

    outer_pcd = o3d.geometry.PointCloud()
    outer_pcd.points = o3d.utility.Vector3dVector(outer_points)
    outer_pcd.paint_uniform_color((1, 0, 0))

    middle_pcd = o3d.geometry.PointCloud()
    middle_pcd.points = o3d.utility.Vector3dVector(middle_points)
    middle_pcd.paint_uniform_color((0, 1, 0))

    inner_pcd = o3d.geometry.PointCloud()
    inner_pcd.points = o3d.utility.Vector3dVector(inner_points)
    inner_pcd.paint_uniform_color((0, 0, 1))

    o3d.visualization.draw_geometries([inner_pcd])
