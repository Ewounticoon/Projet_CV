from calibration.calibration import calibrate_camera
from rectification.StereoRectification_StepbyStep import Rectif_Stereo
from matching.feature_matching import detect_features_sift, match_features_bf, plot_matches
from disparity.disparity import compute_disparity_map, create_height_map
from reconstruction.reconstruction import reconstruct_3D, visualize_point_cloud


import cv2
import numpy as np
import open3d as o3d

def main():
    #img1 = cv2.imread('Taken_photos/photo_left.jpg')
    #img2 = cv2.imread('Taken_photos/photo_right.jpg')
    img1 = cv2.imread('chess1/im0.png')
    img2 = cv2.imread('chess1/im1.png')
    if img1 is None or img2 is None:
        print("Erreur de chargement des images.")
        return

#---- Calibration ----#

    K1,K2,R,T=calibrate_camera()
    R1 = np.eye(3)  # Rotation identité pour la caméra 1
    R2 = R  # Supposons que la caméra 2 est aussi alignée

    # Matrices intrinsèques
    K1 = np.array([[1758.23, 0, 953.34],
                   [0, 1758.23, 552.29],
                   [0, 0, 1]])

    K2 = np.array([[1758.23, 0, 953.34],
                   [0, 1758.23, 552.29],
                   [0, 0, 1]])

    # Rotation : identique pour les deux (parallèles)
    # Rotation identité pour les deux caméras (elles sont déjà alignées)
    R1 = np.eye(3)
    R2 = np.eye(3)

    # Translation : baseline de 111.53 mm sur l’axe X
    T = np.array([111.53, 0, 0])  # En millimètres

    #T = np.array([-15.27075501, -25.26244818, 102.40720507])
    #print(T)

#---- Rectification ----#

    rectif_image=Rectif_Stereo(K1,K2,R1,R2,T)
    h, w, _ = img1.shape
    R_rect=rectif_image.rectified_coord_system()
    H1,H2 = rectif_image.compute_homographie(R_rect)

    # Application des homographies
    img1_rectified = rectif_image.warp_image(img1, H1, (w, h))
    img2_rectified = rectif_image.warp_image(img2, H2, (w, h))
    rectif_image.draw_epipolar_lines(img1_rectified, img2_rectified, num_lines=10)
    img1_rect_rgb = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2RGB)
    img2_rect_rgb = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2RGB)

    # Dimensions de l'image
    R1 = np.eye(3)  # Rotation identité pour la caméra 1
    R2 = R  # Supposons que la caméra 2 est aussi alignée


#---- Reconstruction dense via carte de disparité ----#

    #image_size = (img1.shape[1], img1.shape[0])

    #R1, R2, P1, P2, Q = rectif_image.stereo_rectify(K1, K2, R1, T, image_size)

    Q=rectif_image.compute_Q(K1,111.53)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    imgL = cv2.equalizeHist(gray1)
    imgR = cv2.equalizeHist(gray2)

    disparity_map = compute_disparity_map(imgL, imgR)
    cv2.imshow("Disparity Map (Color)", disparity_map)
    #cv2.imwrite("disparity_map.png", disparity_map)

##################

    height_map = create_height_map(disparity_map, scale=0.5)

##################

    cv2.imshow("Height Map (Color)", height_map)
    #cv2.imwrite("height_map.png", height_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#---- Reconstruction from cloud point 3d ----#
    points_3D = reconstruct_3D(disparity_map, Q)
    color_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

#---- Visualisation du nuage de points ----#
    visualize_point_cloud(points_3D, color_img)




if __name__ == "__main__":
    main()