# disparity/disparity_utils.py
import cv2
import numpy as np

def compute_disparity_map(imgL, imgR):
    """
    Compute the disparity map using Semi-Global Block Matching (SGBM).
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=640,       # Multiple de 16
        blockSize=3,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=-1,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    disparity_map = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disparity_map


def create_height_map(disparity_map, scale=0.5):
    """
    Create a colorized height map from the disparity map for visualization.
    """
    disparity_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)
    height_map = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

    height_map_resized = cv2.resize(
        height_map, (int(height_map.shape[1] * scale), int(height_map.shape[0] * scale))
    )
    return height_map_resized
