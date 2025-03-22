from StereoRectification_StepbyStep import *
from feature_matching import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Matrices intrinsèques des caméras
K1 = np.array([[1758.23, 0, 953.34], 
               [0, 1758.23, 552.29], 
               [0, 0, 1]])

K2 = np.array([[1758.23, 0, 953.34], 
               [0, 1758.23, 552.29], 
               [0, 0, 1]])

# Définition de la baseline et des matrices de rotation
baseline = 111.53
R1 = np.eye(3)  # Rotation identité pour la caméra 1
R2 = np.eye(3)  # Supposons que la caméra 2 est aussi alignée

# Vecteur translation (baseline le long de l'axe x)
T = np.array([baseline, 0, 0])

def main():
    img1 = cv2.imread('chess1/im0.png', 1)
    img2 = cv2.imread('chess1/im1.png', 1)

    # Vérifier si les images ont été chargées correctement
    if img1 is None or img2 is None:
        print("Erreur : Impossible de charger les images.")
        exit()

    # Convertir en RGB pour affichage correct avec matplotlib
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # Dimensions de l'image
    h, w, _ = img1.shape
    R_rect=rectified_coord_system()
    H1,H2 = compute_homographie(R_rect)
    # Application des homographies
    img1_rectified = warp_image(img1, H1, (w, h))
    img2_rectified = warp_image(img2, H2, (w, h))

    imgL=img1_rectified
    imgR=img2_rectified
    # 2. Détection des points-clés et descripteurs (ex : ORB)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)

    # 3. Matching (avec BruteForce + Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 4. Tri des matches par distance
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. Affichage
    matched_img = cv2.drawMatches(imgL, kp1, imgR, kp2, matches[:50], None, flags=2)
    plt.figure(figsize=(20, 10))
    plt.imshow(matched_img)
    plt.axis('off')
    plt.show()


    # 1. Extraire les points à partir des matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 2. Calculer la matrice essentielle
    E, mask = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)

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
    pts1_h = pts1.T
    pts2_h = pts2.T

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]  # homogène -> cartésien
    points_3d = points_3d.T  # (N, 3)



    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1)
    ax.set_title("Nuage de points 3D reconstruit")
    plt.show()

        ### Reconstruction dense via carte de disparité ###

    # Convertir en niveaux de gris pour la stéréo dense
    grayL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)

    # Paramètres du stéréo matcher (SGBM)
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

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Affichage de la carte de disparité
    plt.figure()
    plt.imshow(disparity, cmap='plasma')
    plt.colorbar()
    plt.title("Carte de disparité dense")
    plt.show()

    # Matrice de reprojection Q (à adapter à ta calibration si nécessaire)
    Q = np.array([
        [1, 0, 0, -K1[0, 2]],
        [0, 1, 0, -K1[1, 2]],
        [0, 0, 0, K1[0, 0]],
        [0, 0, -1 / baseline, 0]
    ])

    # Projection en 3D
    points_3d_dense = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0
    output_points = points_3d_dense[mask]
    output_colors = imgL[mask]

    # Affichage du nuage dense
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(output_points[:, 0], output_points[:, 1], output_points[:, 2],
               c=output_colors / 255.0, s=0.5)
    ax.set_title("Nuage de points 3D dense")
    plt.show()




if __name__ == "__main__":
    main()