import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
from os.path import dirname, abspath

from models.pointnet2_lib.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, \
    PointNetFeaturePropagation


class BodyNet(nn.Module):
    def __init__(self, num_occs: int = 2):
        super(BodyNet, self).__init__()
        self.num_occs = num_occs
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128 + 128 + 64,
                                             [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3 + 3 + self.num_occs, mlp=[128, 128])

        self.occ_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, self.num_occs, 1)
        )

    def forward(self, xyz, occs):
        '''
        x: (B, N, 3)
        p: (B, N)
        '''
        B, N, _ = xyz.shape

        xyz = xyz.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
        occs = occs.long()  # (B, N)
        occs_one_hot = F.one_hot(occs, num_classes=self.num_occs).permute(0, 2, 1).float()  # (B,16,N)

        l0_xyz = xyz
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([occs_one_hot, l0_xyz, l0_points], 1), l1_points)

        occ_logits = self.occ_head(l0_points)  # (B, num_occs, N)

        return {
            'occ': occ_logits,
        }


class BodyNetPart(nn.Module):
    def __init__(self,
                 num_classes=2,
                 num_parts=14,
                 use_normal: bool = False,
                 batchnorm=True):
        super(BodyNetPart, self).__init__()

        self.num_parts = num_parts
        self.num_classes = num_classes
        self.use_normal = use_normal
        normal_channel = 3 if self.use_normal else 0

        self.set_abstractions = nn.ModuleList()
        self.feature_propagators = nn.ModuleList()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3 + normal_channel,
                                             [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(256 + 256 + 512 + 512, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 3 + normal_channel, [128, 128, 128])

        # Add part classifier
        self.final_layers = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128) if batchnorm else None,
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_parts, 1)
        )

        self.part_predictors = nn.Sequential(
            nn.Conv1d(128, 128 * self.num_parts, 1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128 * self.num_parts, num_classes * self.num_parts, 1, groups=self.num_parts)
        )

    def forward(self, inputs):
        """
        Args:
            points (torch.Tensor): shape = (batch_size, num_points, 3 + in_features)
                The points to perform segmentation on.
        Returns:
            (torch.Tensor): shape = (batch_size, num_points, num_classes)
                The score of each point being in each class.
                Note: no softmax or logsoftmax will be applied.
        """
        points = inputs  # (B, N, C)
        xyz = inputs.permute(0, 2, 1)  # (B, C, N)

        l0_points = xyz  # (B, C, N)
        l0_xyz = xyz[:, :3, :] if self.use_normal else xyz  # (B, C, N)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # l1_points: (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 256, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 512, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 1024, 16)

        # decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 64)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 128, 256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 64, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)  # (B, 128, N)

        parts = self.final_layers(l0_points).contiguous()
        parts_softmax = F.softmax(parts, dim=1)

        pred = self.part_predictors(l0_points).contiguous()
        weighted_pred = pred.view(pred.shape[0], self.num_classes, self.num_parts, -1) * parts_softmax.unsqueeze(1)
        weighted_pred = weighted_pred.sum(dim=2)

        return {
            'part_labels': parts,  # torch.Size([1, 14, 10000])
            'occ': weighted_pred  # torch.Size([1, 2, 10000])
        }
