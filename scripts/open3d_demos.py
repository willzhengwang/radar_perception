#!/usr/bin/env python

import open3d as o3d
import numpy as np


if __name__ == "__main__":

    # # demo 1:
    # pcd_data = o3d.data.PCDPointCloud()
    # print(
    #     f"Reading pointcloud from file: fragment.pcd stored at {pcd_data.path}")
    # pcd = o3d.io.read_point_cloud(pcd_data.path)
    # print(pcd)
    # # print("Saving pointcloud to file: copy_of_fragment.pcd")
    # # o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
    #
    # o3d.visualization.draw_geometries([pcd])

    # demo 2: visualize a shapenet point cloud
    pc_file = '../data/shapenetcore_subset/02691156/points/1a04e3eab45ca15dd86060f189eb133.pts'
    pc_label = '../data/shapenetcore_subset/02691156/points_label/1a04e3eab45ca15dd86060f189eb133.seg'
    points = np.loadtxt(pc_file, dtype=np.float32, delimiter=' ')
    labels = np.loadtxt(pc_label, dtype=np.int32, delimiter=' ')

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Map label values to colors
    num_labels = int(np.max(labels))
    label_colors = np.random.rand(num_labels + 1, 3)  # Generate random colors for each label
    colors = label_colors[labels.astype(int)]  # Map labels to colors

    # Assign colors to the point cloud based on the labels
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    print("Done")
