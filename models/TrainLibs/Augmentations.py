import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple
from lib.points_mesh_standarization import sv_points

def random_rotation(points: np.ndarray, angle: float = 20, axis: str = 'z') -> np.ndarray:
    '''random points rotation'''
    angle_deg = np.random.uniform(-angle, angle, 3)
    if axis != 'all':
        angle_deg = np.zeros(3)
        if axis == 'x':
            angle_deg[0] = np.random.uniform(-angle, angle)
        elif axis == 'y':
            angle_deg[1] = np.random.uniform(-angle, angle)
        elif axis == 'z':
            angle_deg[2] = np.random.uniform(-angle, angle)
    angle_rad = np.deg2rad(angle_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad[0]), -np.sin(angle_rad[0])],
        [0, np.sin(angle_rad[0]), np.cos(angle_rad[0])]
    ])
    Ry = np.array([
        [np.cos(angle_rad[1]), 0, np.sin(angle_rad[1])],
        [0, 1, 0],
        [-np.sin(angle_rad[1]), 0, np.cos(angle_rad[1])]
    ])
    Rz = np.array([
        [np.cos(angle_rad[2]), -np.sin(angle_rad[2]), 0],
        [np.sin(angle_rad[2]), np.cos(angle_rad[2]), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    return (points @ R.T).astype(np.float32)


def random_isotropic_scale(points: np.ndarray, scale: Tuple[float, float] = (0.8, 1.2),
                           anisotropic: bool = False) -> np.ndarray:
    '''random points scale'''
    if anisotropic:
        scales = np.random.uniform(scale[0], scale[1], 3)
    else:
        scales = np.ones(3) * np.random.uniform(scale[0], scale[1])
    return (points * scales).astype(np.float32)


def random_jitter(points: np.ndarray, noise_std: float = 0.01, clip: float = 0.05) -> np.ndarray:
    '''random points jitter'''
    noise = np.random.normal(0, noise_std, points.shape)
    noise = np.clip(noise, -clip, clip)
    return (points + noise).astype(np.float32)


def random_dropout(points: np.ndarray, max_drop_ratio: float = 0.2) -> np.ndarray:
    '''random points dropout'''
    keep_ratio = 1 - np.random.uniform(0, max_drop_ratio)
    keep_num = int(points.shape[0] * keep_ratio)
    keep_num = max(keep_num, 1)
    idx = np.random.permutation(points.shape[0])[:keep_num]
    return points[idx].astype(np.float32)


def random_translate(points: np.ndarray, max_translation: float = 0.1) -> np.ndarray:
    '''random translate'''
    translation = np.random.uniform(-max_translation, max_translation, 3)
    return (points + translation).astype(np.float32)


class StrictAugmentor:
    def __init__(self,
                 scale: Tuple[float, float] = (0.8, 1.2),
                 anisotropic: bool = False,
                 angle: float = 10.0,
                 axis: str = 'z',
                 noise_std: float = None,
                 max_translation: float = 0.1,
                 dropout_ratio: float = None):
        self.scale_range = scale
        self.anisotropic = anisotropic
        self.max_angle = angle
        self.axis = axis
        self.noise_std = noise_std
        self.max_translation = max_translation
        self.dropout_ratio = dropout_ratio

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if self.max_angle is not None and self.max_angle > 0:
            points = random_rotation(points, self.max_angle, self.axis)

        if self.scale_range is not None and self.scale_range[0] < 1 or self.scale_range[1] > 1:
            points = random_isotropic_scale(points, self.scale_range, self.anisotropic)

        if self.noise_std is not None and self.noise_std > 0:
            points = random_jitter(points, self.noise_std)

        if self.max_translation is not None and self.max_translation > 0:
            points = random_translate(points, self.max_translation)

        if self.dropout_ratio is not None and self.dropout_ratio > 0:
            points = random_dropout(points, self.dropout_ratio)

        return points
