import numpy as np 
import cv2

# Construction of projection matrices
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
#P2 = np.hstack((R, t))
P2 = np.hstack((R, t))
P1 = K1 @ P1
P2 = K2 @ P2

print("P1 = \n",P1)
print("P2 = \n",P2)

# 3D point triangulation
points_4D = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
points_3D = points_4D[:3] / points_4D[3]