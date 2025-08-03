import torch
from torch import nn
from torch.nn import functional as F
from models.pointnet2_lib.pointnet2_utils import farthest_point_sample, PointNetSetAbstractionMsg, \
    PointNetFeaturePropagation


def separate_xyz_and_features(points):
    """Break up a point cloud into position vectors (first 3 dimensions) and feature vectors.

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::

            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }

    Args:
        points (torch.Tensor): shape = (batch_size, num_points, 3 + num_features)
            The point cloud to separate.

    Returns:
        xyz (torch.Tensor): shape = (batch_size, num_points, 3)
            The position vectors of the points.
        features (torch.Tensor|None): shape = (batch_size, num_features, num_points)
            The feature vectors of the points.
            If there are no feature vectors, features will be None.
    """
    assert (len(points.shape) == 3 and points.shape[2] >= 3), (
        'Expected shape of points to be (batch_size, num_points, 3 + num_features), got {}'
        .format(points.shape))

    xyz = points[:, :, 0:3].contiguous()
    features = (points[:, :, 3:].transpose(1, 2).contiguous()
                if points.shape[2] > 3 else None)

    return xyz, features


class TPoseNetPart(nn.Module):
    def __init__(self,
                 num_classes=3,
                 num_parts=14,
                 use_normal: bool = False,
                 batchnorm=True):
        super(TPoseNetPart, self).__init__()

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
            'logits': weighted_pred  # torch.Size([1, 3, 10000]) or torch.Size([1, 2, 10000])
        }


if __name__ == "__main__":
    import numpy as np
    import trimesh

    model = TPoseNetPart().to('cuda')
    file_path = r"F:\CAPE_Dataset_Sampled_10000 - 副本\00032\longshort_ATUsquat\longshort_ATUsquat.000001\longshort_ATUsquat.000001.obj"
    mesh = trimesh.load_mesh(file_path)
    points = mesh.sample(10000)
    inputs = torch.from_numpy(np.array(points)).float().unsqueeze(0).to('cuda')

    out = model(inputs)
    part_labels = out['part_labels']
    correspondences = out['logits']
    print(part_labels.shape, correspondences.shape)
