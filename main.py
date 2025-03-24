from StereoRectification_StepbyStep import Rectif_Stereo
#from feature_matching import *
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import glob


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

objpoints = []
imgpointsL, imgpointsR = [], []

images_left = sorted(glob.glob('Projet_CV/calibration_images/left_*.jpg'))
images_right = sorted(glob.glob('Projet_CV/calibration_images/right_*.jpg'))

for fnameL, fnameR in zip(images_left, images_right):
    imgL, imgR = cv2.imread(fnameL), cv2.imread(fnameR)
    if imgL is None or imgR is None:
        continue
    grayL, grayR = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, (9,7), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (9,7), None)

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria))
        imgpointsR.append(cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria))

if len(objpoints) > 0:
    _, K1, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    _, K2, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
    _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, K1, distL, K2, distR, grayL.shape[::-1], criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    print("Matrice intrinsèque gauche:\n", K1)
    print("Distorsion gauche:\n", distL)
    print("Matrice intrinsèque droite:\n", K2)
    print("Distorsion droite:\n", distR)
    print("Rotation:\n", R)
    print("Translation:\n", T)


## Matrices intrinsèques des caméras
#K1 = np.array([[1758.23, 0, 953.34], 
#               [0, 1758.23, 552.29], 
#               [0, 0, 1]])
#
#K2 = np.array([[1758.23, 0, 953.34], 
#               [0, 1758.23, 552.29], 
#               [0, 0, 1]])

# Définition de la baseline et des matrices de rotation
#baseline=111.53
#
#R1 = np.eye(3)  # Rotation identité pour la caméra 1
#R2 = np.eye(3)  # Supposons que la caméra 2 est aussi alignée
#
## Vecteur translation (baseline le long de l'axe x)
#T = np.array([baseline, 0, 0])


img1 = cv2.imread('chess1/im0.png', 1)
img2 = cv2.imread('chess1/im1.png', 1)

# Vérifier si les images ont été chargées correctement
if img1 is None or img2 is None:
    print("Erreur : Impossible de charger les images.")
    exit()

# Convertir en RGB pour affichage correct avec matplotlib
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

def plot_matches(img1, kp1, img2, kp2, matches, good_matches, title="Feature Matches"):
    img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_good_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_matches)
    plt.title(f"{title} (Before Ratio Test)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_good_matches)
    plt.title(f"{title} (After Ratio Test)")
    plt.axis('off')
    
    plt.show()


# Feature Matching using SIFT
def detect_features_sift(image):
    """
    Detects keypoints and computes descriptors using SIFT (Scale-Invariant Feature Transform).
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    print(f"SIFT: {len(keypoints)} keypoints detected.")
    return keypoints, descriptors

# Using the brute force matcher
def match_features_bf(desc1, desc2, method='sift'):
    """
    Matches feature descriptors using Brute-Force Matcher with optional ratio test.
    """
    if desc1 is None or desc2 is None:
        print("Descriptors missing, skipping matching.")
        return [], []
    
    if method == 'sift':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"{method.upper()} BF Matcher: {len(matches)} matches found, {len(good_matches)} after ratio test.")
    return matches, good_matches



# 🔹 Fonction pour calculer la carte de disparité
def compute_disparity_map(imgL, imgR):
    """
    Compute disparity map from stereo images.
    """

# Parameter for the bike :
#    stereo = cv2.StereoSGBM_create(
#        minDisparity=1,
#        numDisparities=16*16,  # Capturer plus de profondeur
#        blockSize=15,  # Capturer les petits détails
#        P1=8 * 3 * 8 ** 2,  
#        P2=32 * 3 * 8 ** 2,  
#        disp12MaxDiff=2,  # Désactive la vérification stricte
#        uniquenessRatio=1,  # Permet plus de correspondances
#        speckleWindowSize=100,  
#        speckleRange=1,  
#        preFilterCap=63,  # Améliore la détection des contrastes
#        mode=cv2.STEREO_SGBM_MODE_HH  # Mode haute précision
#    )
#

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=640,  # Capturer plus de profondeur
        blockSize=3,  # Capturer les petits détails
        P1=8 * 3 * 5 ** 2,  
        P2=32 * 3 * 5 ** 2,  
        disp12MaxDiff=-1,  # Désactive la vérification stricte
        uniquenessRatio=5,  # Permet plus de correspondances
        speckleWindowSize=50,  
        speckleRange=2,  
        preFilterCap=63,  # Améliore la détection des contrastes
        mode=cv2.STEREO_SGBM_MODE_HH  # Mode haute précision
    )
    disparity_map = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disparity_map

# Fonction pour générer une carte de hauteur colorée
def create_height_map(disparity_map, scale=0.5):
    """
    Create color hight map from disparity.
    """
    disparity_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)
    height_map = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    height_map_resized = cv2.resize(
        height_map, (int(height_map.shape[1] * scale), int(height_map.shape[0] * scale))
    )
    return height_map_resized


# function to reconstruc 3D points
def reconstruct_3D(disparity_map, Q):
    """
    Reconstruct 3D points from disparity map.
    """
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    return points_3D


# Fonction pour afficher le nuage de points en 3D
def visualize_point_cloud(points_3D, colors):
    """
    Visualize the 3D point cloud.
    :param points_3D: Tableau de points 3D (taille H x W x 3).
    :param colors: Image couleur associée (taille H x W x 3).
    """
    # Filtrage des points valides : éliminer les points avec des valeurs aberrantes en Z
    mask = (points_3D[:, :, 2] < 10000) & (points_3D[:, :, 2] > -10000)
    points = points_3D[mask]
    colors = colors[mask]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3) / 255.0)
    o3d.visualization.draw_geometries([point_cloud])

def main():
    # Dimensions de l'image
    R1 = np.eye(3)  # Rotation identité pour la caméra 1
    R2 = np.eye(3)  # Supposons que la caméra 2 est aussi alignée
    rectif_image=Rectif_Stereo(K1,K2,R1,R2,T)
    h, w, _ = img1.shape
    R_rect=rectif_image.rectified_coord_system()
    H1,H2 = rectif_image.compute_homographie(R_rect)

    # Application des homographies
    img1_rectified = rectif_image.warp_image(img1, H1, (w, h))
    img2_rectified = rectif_image.warp_image(img2, H2, (w, h))


###### FEATURE MATCHING #######
    keypoints1, descriptors1 = detect_features_sift(img1_rectified)
    keypoints2, descriptors2 = detect_features_sift(img2_rectified)

    matches_bf_sift, good_matches_bf_sift = match_features_bf(descriptors1, descriptors2, 'sift')
    # Extraction of corresponding points using brute-force matcher
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches_bf_sift])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches_bf_sift])

    plot_matches(img1_rectified, keypoints1, img2_rectified, keypoints2, matches_bf_sift, good_matches_bf_sift, "Brute-Force SIFT Matcher")

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



    # 2. Calculer la matrice essentielle
    E, mask = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    inlier_pts1 = pts1[mask.ravel()==1]
    inlier_pts2 = pts2[mask.ravel()==1]

    # 3. Récupérer R (rotation) et t (translation) à partir de E
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K1)

    # Affichage simple
    print("Matrice essentielle E:\n", E)
    print("Rotation R:\n", R)
    print("Translation t:\n", t)

    ## Matrice de projection ##

    # Caméra gauche (référence)
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Caméra droite (position relative connue)
    P2 = K1 @ np.hstack((R, t))

    ## Triangulation ##

    # Les points doivent être (2, N) et float
    pts1_h = inlier_pts1.T
    pts2_h = inlier_pts2.T

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]  # homogène -> cartésien


######################################################################""


    # Visualisation with Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    o3d.visualization.draw_geometries([pcd], window_name="3D Reconstruction", width=800, height=600)



#######################################################################################################
    ### Reconstruction dense via carte de disparité ###
    image_size = (img1.shape[1], img1.shape[0])

    R1, R2, P1, P2, Q = rectif_image.stereo_rectify(K1, K2, R1, T, image_size)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    imgL = cv2.equalizeHist(gray1)
    imgR = cv2.equalizeHist(gray2)

    disparity_map = compute_disparity_map(imgL, imgR)

##################

    height_map = create_height_map(disparity_map, scale=0.5)

##################

    cv2.imshow("Height Map (Color)", height_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Reconstruction from cloud point 3d
    points_3D = reconstruct_3D(disparity_map, Q)

    color_img = cv2.imread("chess1/im0.png")
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)


    if color_img is None:
        print(" Error : Impossible de charger l'image couleur. Vérifiez le chemin.")
        exit(1)

    # Visualisation du nuage de points
    visualize_point_cloud(points_3D, color_img)




if __name__ == "__main__":
    main()