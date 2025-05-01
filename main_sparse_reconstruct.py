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
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

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




#---- Feature Matching ----#

    keypoints1, descriptors1 = detect_features_sift(img1_rgb)
    keypoints2, descriptors2 = detect_features_sift(img2_rgb)

    matches_bf_sift, good_matches_bf_sift = match_features_bf(descriptors1, descriptors2, 'sift')

    plot_matches(img1_rgb, keypoints1, img2_rgb, keypoints2, matches_bf_sift, good_matches_bf_sift, "Brute-Force SIFT Matcher")

    # Extraction of corresponding points using brute-force matcher
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches_bf_sift])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches_bf_sift])


    ## 2. Détection des points-clés et descripteurs (ex : ORB)
    #orb = cv2.ORB_create(nfeatures=1000)
    #kp1, des1 = orb.detectAndCompute(img1_rectified, None)
    #kp2, des2 = orb.detectAndCompute(img2_rectified, None)
#
    ## 3. Matching (avec BruteForce + Hamming)
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #matches = bf.match(des1, des2)
#
    ## 4. Tri des matches par distance
    #matches = sorted(matches, key=lambda x: x.distance)
#
    ## 5. Affichage
    #matched_img = cv2.drawMatches(img1_rectified, kp1, img2_rectified, kp2, matches[:50], None, flags=2)
    #plt.figure(figsize=(20, 10))
    #plt.imshow(matched_img)
    #plt.axis('off')
    #plt.show()
#
##################################################
#
    ## 1. Extraire les points à partir des matches
    #pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    #pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])



    # Calcule de la matrice essentielle
    E, mask = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Récupérer R (rotation) et t (translation) à partir de E
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K1)

    # Affichage simple
    print("Matrice essentielle E:\n", E)
    print("Rotation R:\n", R)
    print("Translation t:\n", t)

#----Matrice de projection ----#

    # Caméra gauche (référence)
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Caméra droite (position relative connue)
    P2 = K1 @ np.hstack((R, t))

    ## Triangulation ##
    inlier_pts1 = pts1[mask.ravel()==1]
    inlier_pts2 = pts2[mask.ravel()==1]
    # Les points doivent être (2, N) et float
    pts1_h = inlier_pts1.T
    pts2_h = inlier_pts2.T

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]  # homogène -> cartésien



#---- Visualisation with Open3D ----#

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    o3d.visualization.draw_geometries([pcd], window_name="3D Reconstruction", width=800, height=600)


if __name__ == "__main__":
    main()