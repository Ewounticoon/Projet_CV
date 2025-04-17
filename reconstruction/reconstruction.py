# reconstruction/pointcloud_utils.py
import cv2
import numpy as np
import open3d as o3d

def reconstruct_3D(disparity_map, Q):
    """
    Reconstruct 3D points from disparity using reprojection matrix Q.
    """
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    return points_3D


def visualize_point_cloud(points_3D, colors):
    """
    Visualize the 3D point cloud with Open3D.
    :param points_3D: 3D points array (H x W x 3)
    :param colors: Corresponding color image (H x W x 3)
    """
    # Filtrer les points valides
    mask = (points_3D[:, :, 2] < 10000) & (points_3D[:, :, 2] > -10000)
    points = points_3D[mask]
    colors = colors[mask]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3) / 255.0)

    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud 3D")
