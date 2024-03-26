#!/usr/bin/env python
"""
PointNet++
Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
from typing import List, Optional
import torch
from torch import nn
from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import MeanMetric, MaxMetric
from lightning import LightningModule

from src.utils import pylogger
log = pylogger.get_pylogger(__name__)


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate the squared Euclidean distance for every pair of points between two point clouds.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    @param src: (batch_size, N, channels), N is the number of points in the source point cloud
    @param dst: (batch_size, M, channels), M is the number of points in the destination point cloud
    @return:
        dist: [batch_size, N, M], per-point square distance
    """
    B, N, _ = src.shape  # B: batch_size
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # (B, N, M)
    dist += torch.sum(src ** 2, -1).view(B, N, 1)  # (B, N, M)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # (B, N, M)
    return dist


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Indexing points
    @param points: (batch_size, num_points, num_channels)
    @param indices: (batch_size, idx_dim0, idx_dim1, ...)
    @return:
        indexed_points: (batch_size, idx_dim0, idx_dim1, ..., num_channels)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(indices.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indices.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, indices, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, num_centroids: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) for selecting a number of centroids from the input point clouds.
    @param xyz: (batch_size, num_points, num_channels=3) - a batch of point clouds
    @param num_centroids: the number of sampled points (centroids)
    @return:
        centroid_inds: (batch_size, num_samples) of sample point indices
    """
    device = xyz.device
    B, N, C = xyz.shape  # batch_size, num_points, num_channels
    centroid_inds = torch.zeros(B, num_centroids, dtype=torch.long).to(device)

    # Initialize the distances between points to the selected sample points with max inf.
    distance = torch.zeros(B, N).to(device) + float('inf')

    # For each point cloud, randomly select a point as the farthest point
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_inds = torch.arange(B, dtype=torch.long).to(device)

    for i in range(num_centroids):
        # set the new centroid as the last farthest point
        centroid_inds[:, i] = farthest
        centroid = xyz[batch_inds, farthest, :].view(B, 1, C)

        # calculate the distances between the points with the centroid (i.e. the newly selected sample)
        dist = torch.sum((xyz - centroid) ** 2, -1)

        # Keep updating the distance array, which enables the selection of farthest point to all existing centroids.
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, -1)[1]
    return centroid_inds


def query_ball(radius, num_samples, xyz, query_xyz):
    """
    Find sample points within a radius for each query point (centroid).
    @param radius: local region radius
    @param num_samples: max sample number in local region
    @param xyz: (batch_size, num_points, num_channels=3) - a batch of point clouds
    @param query_xyz: query points, [B, S, 3]
    @return:
        group_inds: (batch_size, num_centroids, num_samples) the centroid index of each point
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = query_xyz.shape
    group_inds = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    # calculate the squared distance for every pair of points between two point clouds.
    sqr_dists = square_distance(query_xyz, xyz)  # (batch_size, S, N)
    group_inds[sqr_dists > radius ** 2] = N
    group_inds = group_inds.sort(dim=-1)[0][:, :, :num_samples]

    # if the number of sample points within the radius is < num_samples, fill the remaining with the first point
    group_first = group_inds[:, :, 0].view(B, S, 1).repeat([1, 1, num_samples])
    mask = group_inds == N
    group_inds[mask] = group_first[mask]
    return group_inds


def sample_and_group(num_centroids: int, radius: float, num_samples: int, points_xyz: torch.Tensor,
                     points_features: Optional[torch.Tensor] = None, return_fps: bool = False):
    """
    FPS sampling + Grouping for a batch of point clouds.
    @param num_centroids: number of centroids in FPS
    @param radius: local region radius
    @param num_samples: max sample number in a local region
    @param points_xyz: (batch_size, num_points, 3) - coordinates of point clouds
    @param points_features: (batch_size, num_points, D) - features of point clouds
    @param return_fps: True - return FPS results. False - return
    @return:
        centroids_xyz: (batch_size, num_centroids, 3)
        new_points: (batch_size, num_centroids, num_samples, C+D)

    """
    B, N, C = points_xyz.shape
    S = num_centroids

    # find centroids of local regions
    fps_inds = farthest_point_sample(points_xyz, num_centroids)  # [B, num_centroids, C]

    # centroid points
    centroids_xyz = index_points(points_xyz, fps_inds)

    # query a fixed number of samples for each centroid
    group_inds = query_ball(radius, num_samples, points_xyz, centroids_xyz)  # (batch_size, num_centroids, num_samples)

    # each point in the xyz has a centroid label
    grouped_xyz = index_points(points_xyz, group_inds)  # [B, num_centroids, num_samples, C]

    # centralize the coordinates of the points by subtracting the local centroids
    grouped_xyz_norm = grouped_xyz - centroids_xyz.view(B, S, 1, C)

    if points_features is not None:
        grouped_points = index_points(points_features, group_inds)
        # concatenate the xyz coordinates and features of the point clouds
        # [batch_size, num_centroids, num_samples, C+D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if return_fps:
        return centroids_xyz, new_points, grouped_xyz, fps_inds
    else:
        return centroids_xyz, new_points


def sample_and_group_all(points_xyz, points_features):
    """
    Sample and group all points into one local region. There is only one centroid.

    @param points_xyz: (batch_size, num_points, 3) - coordinates of point clouds
    @param points_features: (batch_size, num_points, D) - features of point clouds
    @return:
        centroid_xyz: (batch_size, 1, 3)
        new_points:  (batch_size, 1, num_samples, C+D)
    """
    device = points_xyz.device
    B, N, C = points_xyz.shape
    centroid_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = points_xyz.view(B, 1, N, C)
    if points_features is not None:
        new_points = torch.cat([grouped_xyz, points_features.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return centroid_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    A hierarchical feature learning framework that aims to capture local context at different scales.
    A set abstraction block is made of three key layers: Sampling layer, Grouping layer and PointNet layer.
    Sampling layer: use iterative farthest point sampling (FPS) to choose a subset of query points (centroids).
    Grouping layer: use Ball Query to finds all points that are within a radius to the query points (centroids).
    PointNet layer: for local pattern learning.
    """

    def __init__(self, num_centroids: Optional[int], radius: Optional[float], num_samples: Optional[int],
                 in_channels: int, mlp_channels: List[int], group_all: bool):
        super().__init__()
        self.num_centroids = num_centroids
        self.radius = radius
        self.num_samples = num_samples
        self.in_channels = in_channels

        self.mlp = nn.ModuleList()
        for out_channels in mlp_channels:
            self.mlp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels
        self.group_all = group_all

    def forward(self, points_xyz, points_features):
        """
        @param points_xyz: (B, C, N), i.e. (batch_size, num_coordinates, num_points) - coordinates of point clouds
        @param points_features: (B, D, N) i.e. (batch_size, num_features, num_points) - features of point clouds
        @return: 
        """
        points_xyz = points_xyz.permute(0, 2, 1)  # (batch_size, num_points, 3). 3=num_coordinates
        if points_features is not None:
            points_features = points_features.permute(0, 2, 1)  # (batch_size, num_points, num_features)

        if self.group_all:
            centroid_xyz, new_points = sample_and_group_all(points_xyz, points_features)
        else:
            centroid_xyz, new_points = sample_and_group(self.num_centroids, self.radius, self.num_samples,
                                                        points_xyz, points_features)
        # centroid_xyz: (batch_size, num_samples, 3)
        # new_points: (batch_size, num_centroids, num_samples, 3 + num_features)
        new_points = new_points.permute(0, 3, 2, 1)  # (batch_size, 3+num_features, num_samples, num_centroids)

        for layer in self.mlp:
            new_points = layer(new_points)
        # new_points: (batch_size, mlp_channels[-1], num_samples, num_centroids)
        # extract the global features of each local region (centroid) == local features
        new_points = torch.max(new_points, 2)[0]
        centroid_xyz = centroid_xyz.permute(0, 2, 1)  # (batch_size, 3, num_samples)
        return centroid_xyz, new_points


class PointNetSetAbstractionMSG(nn.Module):
    """
    Multi-scale Grouping (MSG).
    A simple but effective way to capture multiscale patterns is to apply grouping layers with different scales
    followed by according PointNets to extract features of each scale.
    Features at different scales are concatenated to form a multi-scale feature.
    """
    def __init__(self, num_centroids: int, radius_list: List[float], num_samples_list: List[int], in_channels: int,
                 mlp_channels_list: List[List[int]]):
        # 512, [0.1, 0.2, 0.4], [16, 32, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        super().__init__()
        self.num_centroids = num_centroids
        self.radius_list = radius_list
        self.num_samples_list = num_samples_list
        self.mlp_channels_list = mlp_channels_list
        self.mlp_blocks = nn.ModuleList()
        for i, mlp_channels in enumerate(mlp_channels_list):
            mlp = nn.ModuleList()
            # In different scales, the first input channels are the same
            last_channels = in_channels
            for out_channels in mlp_channels:
                mlp.append(
                    nn.Sequential(
                        nn.Conv2d(last_channels, out_channels, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    )
                )
                last_channels = out_channels
            self.mlp_blocks.append(mlp)

    def forward(self, points_xyz, points_features):
        """
        @param points_xyz: (B, C, N), i.e. (batch_size, num_coordinates, num_points) - coordinates of point clouds
        @param points_features: (B, D, N) i.e. (batch_size, num_features, num_points) - features of point clouds
        @return:
        """
        points_xyz = points_xyz.permute(0, 2, 1)  # (batch_size, num_points, 3). 3=num_coordinates
        if points_features is not None:
            points_features = points_features.permute(0, 2, 1)  # (batch_size, num_points, num_features)

        B, N, C = points_xyz.shape  # batch_size, num_points, num_coordinates
        S = self.num_centroids

        # Sampling: select/sample a number of centroids with FPS
        centroid_inds = farthest_point_sample(points_xyz, self.num_centroids)

        # Get the xyz coordinate of the centroids
        centroids_xyz = index_points(points_xyz, centroid_inds)  # new_xyz

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.num_samples_list[i]  # num_samples in a local region

            # Grouping: sample K points from a local region of each centroid
            # Each point is marked with a centroid index
            group_inds = query_ball(radius, K, points_xyz, centroids_xyz)
            grouped_xyz = index_points(points_xyz, group_inds)  # [B, num_centroids, num_samples, C]

            # Centralize the coordinates of each point to its associated centroid
            grouped_xyz -= centroids_xyz.view(B, S, 1, C)

            if points_features is not None:
                grouped_features = index_points(points_features, group_inds)
                grouped_points = torch.cat([grouped_features, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C+D, K, S]

            # Get the global features for each local region
            for layer in self.mlp_blocks[i]:
                grouped_points = layer(grouped_points)

            # Get the final global features
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        centroids_xyz = centroids_xyz.permute(0, 2, 1)
        # Concatenate features at different scales to form a multi-scale feature
        return centroids_xyz, torch.cat(new_points_list, dim=1)


class PointNetFeaturePropagation(nn.Module):
    """
    In PointNetSetAbstraction (PointNetSetAbstractionMSG) layer, the original point set is subsampled.
    However, in a segmentation task, we want to obtain point features for all the original points.

    PointNetFeaturePropagation aims to propagate features from subsampled points to the original points.
    It's achieved by inverse distance weighted interpolation.
    """
    def __init__(self, in_channels: int, mlp_channels: List[int]):
        super().__init__()
        self.mlp = nn.ModuleList()
        for out_channels in mlp_channels:
            self.mlp.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels

    def forward(self, original_xyz, sampled_xyz, original_features, sampled_features):
        """
        Upsample / Interpolate features of points.

        @param original_xyz: (B, 3, num_points), Coordinates of the original points where sampled_xyz are sampled from.
        @param sampled_xyz: (B, 3, num_centroids), Coordinates of the downsampled points.
        @param original_features: (B, num_original_features, num_points), Features of the original points.
        @param sampled_features: (B, num__sampled_features, num_centroids),  Features of the downsampled points.
        @return:
            new_features: (B, num_original_features+num__sampled_features, num_points).
        """
        original_xyz = original_xyz.permute(0, 2, 1)
        sampled_xyz = sampled_xyz.permute(0, 2, 1)

        sampled_features = sampled_features.permute(0, 2, 1)

        B, N, C = original_xyz.shape
        _, S, _ = sampled_xyz.shape

        if S == 1:
            interp_features = sampled_features.repeat(1, N, 1)
        else:
            # interpolating features by means of inverse distance weighted average
            dists = square_distance(original_xyz, sampled_xyz)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interp_features = torch.sum(index_points(sampled_features, idx) * weight.view(B, N, 3, 1), dim=2)

        if original_features is not None:
            original_features = original_features.permute(0, 2, 1)
            new_features = torch.cat([original_features, interp_features], dim=-1)
        else:
            new_features = interp_features

        # Apply mlp to concatenated features.
        new_features = new_features.permute(0, 2, 1)  # (B, num_channels, num_points)
        for layer in self.mlp:
            new_features = layer(new_features)

        return new_features


class PointNet2SSGCls(nn.Module):
    """
    PointNet++ with SSG Single-scale grouping) for classification
    """
    def __init__(self, num_classes, with_normals=True):
        """
        Init function
        @param num_classes: number of classes
        @param with_normals: True - input point clouds with additional normal vectors.
                               False: input only coordinates
        """
        super().__init__()
        in_channels = 6 if with_normals else 3
        self.num_classes = num_classes
        self.with_normals = with_normals
        # self, num_centroids: int, radius: float, num_samples: int, in_channels: int,
        #                  mlp_channels: List[int], group_all: bool
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channels, [64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, points: torch.Tensor):
        """
        @param points: [batch_size, num_channels, num_points] of point clouds.
        The first 3 channels are (x, y, z) coordinates. The later are features such as normal vector.
        @return:
        """
        B, _, _ = points.shape
        xyz = points[:, :3, :]
        if self.with_normals and points.shape[1] > 3:
            norm = points[:, 3:, :]
        else:
            norm = None
        l1_xyz, l1_features = self.sa1(xyz, norm)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        x = l3_features.view(B, 1024)
        x = self.drop1(nn.functional.relu(self.bn1(self.fc1(x))))
        x = self.drop2(nn.functional.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)  # x == logits, use entropy_loss
        # x = nn.functional.log_softmax(x, -1)  # use nll_loss
        return x, l3_features


class PointNet2MSGCls(nn.Module):
    """
    PointNet++ with MSG (multi-scale grouping) for classification
    """
    def __init__(self, num_classes: int, with_normals: bool = True):
        """
        @param num_classes: number of classes
        @param with_normals: Ture - point clouds have coordinates + normals.
                             False - point clouds have coordinates only.
        """
        super().__init__()
        in_channels = 6 if with_normals else 3
        self.num_classes = num_classes
        self.with_normals = with_normals
        # Note that the change trend of num_centroids, radius, num_samples, and mlp_channels is very similar to CNN on
        # images. Basically, the depth (number of channels) increases with the increase of receptive field, and with
        # the decrease of the spatial resolution (number of centroids).
        self.sa1 = PointNetSetAbstractionMSG(512,
                                             [0.1, 0.2, 0.4],
                                             [16, 32, 128],
                                             in_channels,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMSG(128,
                                             [0.2, 0.4, 0.8],
                                             [32, 64, 128],
                                             320 + 3,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None,
                                          640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, points):
        """
        @param points: [batch_size, num_channels, num_points] of point clouds.
        The first 3 channels are (x, y, z) coordinates. The later are features such as normals.
        @return:
        """
        B, _, _ = points.shape
        xyz = points[:, :3, :]
        if self.with_normals and points.shape[1] > 3:
            norm = points[:, 3:, :]
        else:
            norm = None
        l1_xyz, l1_features = self.sa1(xyz, norm)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        x = l3_features.view(B, 1024)
        x = self.drop1(nn.functional.relu(self.bn1(self.fc1(x))))
        x = self.drop2(nn.functional.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)  # x == logits, use entropy_loss
        # x = nn.functional.log_softmax(x, -1)  # use nll_loss
        return x, l3_features


class PointNet2SSGPartSeg(nn.Module):
    """
    PointNet++ with SSG (single-scale grouping) for part segmentation
    """
    def __init__(self, num_classes: int, with_normals=True):
        super().__init__()

        self.num_classes = num_classes
        self.with_normals = with_normals
        in_channels = 6 if with_normals else 3
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channels,
                                          [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3,
                                          [128, 128, 256], False)
        # sa3 group all into one centroid
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3,
                                          [256, 512, 1024], True)

        self.fp3 = PointNetFeaturePropagation(256 + 1024, [256, 256])
        self.fp2 = PointNetFeaturePropagation(128 + 256, [256, 128])

        # in fp1, the in_channels: 3*2: dimension_of_coords + dimension_of_normals. This is different from
        # https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_part_seg_ssg.py
        self.fp1 = PointNetFeaturePropagation(128 + 3 + 3, [128, 128, 128])
        # FC layers for segmentation
        self.fc1 = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=True),  # bias can be set as False as it's followed by a BN
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
        )
        self.conv = nn.Conv1d(128, num_classes, 1)

    def forward(self, points: torch.Tensor):
        """
        @param points: [batch_size, num_channels, num_points] of point clouds.
        The first 3 channels are (x, y, z) coordinates. The later are features such as normals.
        # @param seg_labels: [batch_size, 1, num_points] segmentation labels
        @return:
        """
        B, _, N = points.shape
        l0_xyz = points[:, :3, :]
        if self.with_normals and points.shape[1] > 3:
            l0_features = points[:, 3:, :]
        else:
            l0_features = None
        l1_xyz, l1_features = self.sa1(l0_xyz, l0_features)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)

        # Feature propagation
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)

        # The seg_labels are used as features and concatenated with coords and normals in the implementation:
        # https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_part_seg_ssg.py
        l0_features = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_features], dim=1), l1_features)

        x = self.fc1(l0_features)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # logits, use entropy_loss
        return x


class PointNet2MSGPartSeg(nn.Module):
    """
    PointNet++ with MSG (multi-scale grouping) for Part Segmentation
    """
    def __init__(self, num_classes: int, with_normals: bool = True):
        """
        @param num_classes: number of classes
        @param with_normals: Ture - point clouds have coordinates + normals.
                             False - point clouds have coordinates only.
        """
        super().__init__()
        in_channels = 6 if with_normals else 3
        self.num_classes = num_classes
        self.with_normals = with_normals
        # Note that the change trend of num_centroids, radius, num_samples, and mlp_channels is very similar to CNN on
        # images. Basically, the depth (number of channels) increases with the increase of receptive field, and with
        # the decrease of the spatial resolution (number of centroids).
        self.sa1 = PointNetSetAbstractionMSG(512,
                                             [0.1, 0.2, 0.4],
                                             [32, 64, 128],
                                             in_channels,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMSG(128,
                                             [0.4, 0.8],
                                             [64, 128],
                                             320 + 3,  # 64+128+128=320
                                             [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None,
                                          512 + 3,  # 256+256=512
                                          [256, 512, 1024], True)

        self.fp3 = PointNetFeaturePropagation(256 + 256 + 1024, [256, 256])
        self.fp2 = PointNetFeaturePropagation(64 + 128 + 128 + 256, [256, 128])

        self.fp1 = PointNetFeaturePropagation(128 + 3 + 3, [128, 128])

        # FC layers for segmentation
        self.fc1 = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=True),  # bias can be set as False as it's followed by a BN
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
        )
        self.conv = nn.Conv1d(128, num_classes, 1)

    def forward(self, points):
        """
        @param points: [batch_size, num_channels, num_points] of point clouds.
        The first 3 channels are (x, y, z) coordinates. The later are features such as normals.
        @return:
        """
        B, _, _ = points.shape
        l0_xyz = points[:, :3, :]
        if self.with_normals and points.shape[1] > 3:
            l0_features = points[:, 3:, :]  # normals
        else:
            l0_features = None
        l1_xyz, l1_features = self.sa1(l0_xyz, l0_features)  # l1_features: B, (64+128+128)=320, num_centroids=512
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)  # l2_features: B, (256+256)=512, num_centroids=128
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)  # l3_features: B, 1024, num_centroids=1

        # Feature Propagation layers
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)  # l2_features: B, 256, num_centroids=128
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)  # l1_features: B, 128, num_centroids=512
        l0_features = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_features], dim=1), l1_features)

        x = self.fc1(l0_features)
        x = self.conv(x)  # x == logits, use entropy_loss
        # x = nn.functional.log_softmax(x, -1)  # use nll_loss
        x = x.permute(0, 2, 1)
        return x


class PointNet2ClsModule(LightningModule):
    """
    PointNet++ classification - Lightning Module
    """
    def __init__(self, net: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
                 compile: bool = False):
        """
        Init function of the LightningModule
        @param net: The model to train. Either PointNet2MSGCls or PointNet2SSGCls.
        @param optimizer: The optimizer to use for training.
        @param scheduler: The learning rate scheduler to use for training.
        @param compile: True - compile model for faster training with pytorch 2.0.
        """
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt.
        # logger=True: send the hyperparameters to the logger.
        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def setup(self, stage: str) -> None:
        """
        Lightning hook that is called at the beginning of fit (train + validate),
        validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        @param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins
        """
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch):
        """
        Forward + Loss + Predict a batch of data
        """
        points, labels = batch
        points = points.permute(0, 2, 1)  # (batch_size, 3+num_features, num_point)

        logits, _ = self.forward(points)  # logits: batch_size * num_classes
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: tuple, batch_idx: int):
        """
        Perform a single training step on a batch of data from the training set.

        @param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        @param batch_idx: The index of the current batch.
        @return: A tensor of losses between model predictions and targets.
        """
        loss, preds, labels = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, labels)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step on a batch of data from the validation set
        """
        loss, preds, labels = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, labels)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step on a batch of data from the test set.
        """
        loss, preds, labels = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, labels)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        @return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        return {'optimizer': optimizer}


class PointNet2PartSegModule(LightningModule):
    """
    Lightning Module of PointNet++ for Part Segmentation on Point Clouds
    """
    def __init__(self, net: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
                 compile: bool = False):
        """
        Init function of the LightningModule
        @param net: The model to train. Either PointNet2MSGCls or PointNet2SSGCls.
        @param optimizer: The optimizer to use for training.
        @param scheduler: The learning rate scheduler to use for training.
        @param compile: True - compile model for faster training with pytorch 2.0.
        """
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt.
        # logger=True: send the hyperparameters to the logger.
        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for instance average IoU (intersection over union)
        self.train_iou = MeanMetric()
        self.val_iou = MeanMetric()
        self.test_iou = MeanMetric()
        self.test_ious = []

        # for tracking best so far validation accuracy
        self.val_iou_best = MaxMetric()

    def setup(self, stage: str) -> None:
        """
        Lightning hook that is called at the beginning of fit (train + validate),
        validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        @param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins
        """
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_iou_best.reset()
        self.val_iou.reset()

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch):
        """
        Forward + Loss + Predict a batch of data
        """
        points, seg_labels, encoded_segments = batch
        points = points.permute(0, 2, 1)  # (batch_size, 3+num_features, num_point)
        logits = self.forward(points)  # logits: batch_size * num_points * num_classes

        # calculate instance IoU
        preds = torch.argmax(logits, dim=-1)
        instance_ious = []
        for i in range(points.shape[0]):
            # decode the segments. an encoded str of segments: '12,13,14,15'
            segments = [int(s) for s in encoded_segments[i].split(',')]
            for segment in segments:
                total_union = torch.sum((preds[i] == segment) | (seg_labels[i] == segment))
                if total_union == 0:
                    instance_ious.append(1.0)
                else:
                    intersection = torch.sum((preds[i] == segment) & (seg_labels[i] == segment))
                    instance_ious.append((intersection / total_union).item())

        logits = logits.reshape(-1, self.net.num_classes)
        labels = seg_labels.view(-1, 1)[:, 0]
        loss = self.criterion(logits, labels)
        return loss, instance_ious

    def training_step(self, batch: tuple, batch_idx: int):
        """
        Perform a single training step on a batch of data from the training set.

        @param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        @param batch_idx: The index of the current batch.
        @return: A tensor of losses between model predictions and targets.
        """
        loss, instance_ious = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_iou(torch.mean(torch.tensor(instance_ious)))

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step on a batch of data from the validation set
        """
        loss, instance_ious = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_iou(torch.mean(torch.tensor(instance_ious)))

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step on a batch of data from the test set.
        """
        loss, instance_ious = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_iou(torch.mean(torch.tensor(instance_ious)))
        self.test_ious.extend(instance_ious)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_end(self) -> None:
        # calculate instance average iou
        ave_iou = torch.mean(torch.tensor(self.test_ious))
        log.info("Final instance average iou on the test dataset: {:.2f}".format(ave_iou))

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        @return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        return {'optimizer': optimizer}
