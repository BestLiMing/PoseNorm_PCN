import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
from os.path import dirname, abspath

project_root = dirname(abspath(__file__))
sys.path.append(project_root)

from pointnet2_lib.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, \
    PointNetFeaturePropagation


class FrontToBackNetPart(nn.Module):
    def __init__(self, num_parts: int = 14, use_normal: bool = False):
        super(FrontToBackNetPart, self).__init__()
        self.num_parts = num_parts
        self.use_normal = use_normal
        normal_channel = 3 if self.use_normal else 0

        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3 + normal_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128 + 128 + 64,
                                             [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 256 + 64, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3 + 3 + normal_channel, mlp=[128, 128])

        self.sem = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.num_parts, 1)
        )

        self.offset = nn.Sequential(
            nn.Conv1d(128, 128 * self.num_parts, 1),
            nn.BatchNorm1d(128 * self.num_parts),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128 * self.num_parts, self.num_parts, 1, groups=self.num_parts)
            # nn.Conv1d(128 * self.num_classes, 3 * self.num_classes, 1, groups=self.num_classes)
        )

    def forward(self, inputs):
        points = inputs
        xyz = inputs.permute(0, 2, 1)

        l0_points = xyz
        l0_xyz = xyz[:, :3, :] if self.use_normal else xyz

        # encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B,256,256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B,128,1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)  # (B,128,N)

        # sem
        sem_logits = self.sem(l0_points).contiguous()  # (B, C, N) (B, 14, N)
        sem_probs = F.softmax(sem_logits, dim=1)  # (B, C, N) (B, 14, N)

        # offset
        offset_z = self.offset(l0_points).contiguous()  # (B, C, N) (B, 14, N)
        # weighted_z = offset_z.view(offset_z.shape[0], 3, self.num_classes, -1) * sem_probs.unsqueeze(1)  # (B, 3, C, N) * (B, 1, C, N) -> (B, 3, C, N)
        weighted_z = offset_z * sem_probs  # (B, C, N) (B, 14, N)
        final_z_offset = weighted_z.sum(dim=1)  # (B, N)

        back_points = points.clone()  # (B, N, 3)
        # z_offset = final_z_offset[:, 2, :].transpose(1, 2)  # (B, N, 1)
        back_points[:, :, 2] += final_z_offset  # (B, N, C) (B, N, 3)

        return {
            'part_labels': sem_logits,  # (B, 14, N)
            'back_points': back_points  # (B, N, 3)
        }


class FrontToBackNetPart_NoWeighting(nn.Module):
    def __init__(self, num_parts: int = 14, use_normal: bool = False):
        super(FrontToBackNetPart_NoWeighting, self).__init__()
        self.num_parts = num_parts
        self.use_normal = use_normal
        normal_channel = 3 if self.use_normal else 0

        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3 + normal_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128 + 128 + 64,
                                             [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 256 + 64, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3 + 3 + normal_channel, mlp=[128, 128])

        # self.sem = nn.Sequential(
        #     nn.Conv1d(128, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(128, self.num_classes, 1)
        # )

        self.offset = nn.Sequential(
            nn.Conv1d(128, 128 * self.num_parts, 1),
            nn.BatchNorm1d(128 * self.num_parts),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128 * self.num_parts, self.num_parts, 1, groups=self.num_parts)
            # nn.Conv1d(128 * self.num_classes, 3 * self.num_classes, 1, groups=self.num_classes)
        )

    def forward(self, inputs):
        points = inputs
        xyz = inputs.permute(0, 2, 1)

        l0_points = xyz
        l0_xyz = xyz[:, :3, :] if self.use_normal else xyz

        # encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B,256,256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B,128,1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)  # (B,128,N)

        # sem
        # sem_logits = self.sem(l0_points).contiguous()  # (B, C, N) (B, 14, N)
        # sem_probs = F.softmax(sem_logits, dim=1)  # (B, C, N) (B, 14, N)

        # offset
        offset_z = self.offset(l0_points).contiguous()  # (B, C, N) (B, 14, N)
        # weighted_z = offset_z.view(offset_z.shape[0], 3, self.num_classes, -1) * sem_probs.unsqueeze(1)  # (B, 3, C, N) * (B, 1, C, N) -> (B, 3, C, N)
        weighted_z = offset_z  # * sem_probs  # (B, C, N) (B, 14, N)
        final_z_offset = weighted_z.sum(dim=1)  # (B, N)

        back_points = points.clone()  # (B, N, 3)
        # z_offset = final_z_offset[:, 2, :].transpose(1, 2)  # (B, N, 1)
        back_points[:, :, 2] += final_z_offset  # (B, N, C) (B, N, 3)

        return {
            'back_points': back_points,
        }


def create_model(mode, num_classes: int = 14, use_normal: bool = False):
    if mode == 'front2back_part':
        model = FrontToBackNetPart(num_classes=num_classes, use_normal=use_normal)
    elif mode == 'front2back_part_no_weighting':
        model = FrontToBackNetPart_NoWeighting(num_classes=num_classes, use_normal=use_normal)
    else:
        raise ValueError(f"Invalid loss mode: {mode}")
    return model


if __name__ == "__main__":
    model = FrontToBackNetPart().to('cuda')
    file_path = r"F:\cape_data\00032\longshort_ATUsquat\longshort_ATUsquat.000002\front_views\outside.npz"
    data = np.load(file_path)
    inputs = torch.from_numpy(data['front_points']).float().unsqueeze(0).contiguous().to('cuda')
    targets = torch.from_numpy(data['back_points']).float().unsqueeze(0).contiguous().to('cuda')
    labels = torch.from_numpy(data['labels']).squeeze(-1).long().unsqueeze(0).to('cuda')
    print(inputs.shape, targets.shape, labels.shape)

    sem, back = model(inputs)
    print(sem.shape, back.shape)
    cd_loss = (F.mse_loss(back, targets)) * 1000
    ce_loss = F.cross_entropy(sem, labels)
    total_loss = cd_loss + ce_loss
    print(cd_loss, ce_loss, total_loss)

    assert torch.isfinite(sem).all(), "sem_logits 包含 NaN/Inf"
    assert torch.isfinite(back).all(), "pred_points 包含 NaN/Inf"
    #
    # print(sem.shape)  # 应输出 torch.Size([2, 14, 2048])
    # print(offset.shape)  # 应输出 torch.Size([2, 3, 2048])
