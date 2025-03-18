import cv2
import numpy as np
import matplotlib.pyplot as plt


cam0=np.array([
        [1758.23, 0, 953.34], 
        [0, 1758.23, 552.29], 
        [0, 0, 1]
    ])

cam1=np.array([
        [1758.23, 0, 953.34], 
        [0, 1758.23, 552.29], 
        [0, 0, 1]
    ])

doffs=0
baseline=111.53
width=1920
height=1080
ndisp=290
vmin=75
vmax=262

fx1 = cam0[0][0] 
fy1 = cam0[1][1] 
cx1 = cam0[0][2] 
cy1 = cam0[1][2]  

fx2 = cam1[0][0]  
fy2 = cam1[1][1]
cx2 = cam1[0][2]
cy2 = cam1[1][2]


# Charger les matrices intrinsèques et d'extrinsèques des caméras
K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
K2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
R = np.eye(3)  # Rotation entre les caméras
T = np.array([[baseline], [0], [0]])  # Translation

# Charger les images stéréo
img1 = cv2.imread('Projet_CV/chess1/im0.png', 1)
img2 = cv2.imread('Projet_CV/chess1/im1.png', 1)

# Pour éviter les erreurs de chargement images.
if img1 is None or img2 is None:
    print("Erreur : Impossible de charger les images.")
    exit(1)


h, w,_ = img1.shape

# Dessiner les lignes épipolaires
def draw_epipolar_lines(img, color=(0, 255, 0)):
    """ Dessine des lignes épipolaires sur l'image """
    step = 50  # Espacement entre les lignes
    for y in range(0, img.shape[0], step):
        cv2.line(img, (0, y), (img.shape[1], y), color, 1)
    return img

# Visualiser les lignes épipolaires avant
img1_lines = draw_epipolar_lines(img1.copy())
img2_lines = draw_epipolar_lines(img2.copy())


# Calculer la rectification : 
# cv2.stereoRectify() est utilisée pour calculer les matrices permettant de rectifier les images stéréo. 
# Elle transforme les deux images de manière à aligner leurs lignes épipolaires horizontalement.

# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
#     cameraMatrix1, distCoeffs1,  # Matrice intrinsèque et distorsion de la 1ère caméra
#     cameraMatrix2, distCoeffs2,  # Matrice intrinsèque et distorsion de la 2ème caméra
#     imageSize,  # Taille des images (largeur, hauteur)
#     R, T,  # Rotation et translation entre les caméras
#     flags=cv2.CALIB_ZERO_DISPARITY,  # Mode de rectification
#     alpha=0,  # Facteur de redimensionnement des images rectifiées
# )

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, None, K2, None, (w,h), R, T)

# R1, R2 : Matrices de rotation pour aligner les images gauche et droite.
# P1, P2 : Matrices de projection pour les nouvelles images rectifiées.
# Q : Matrice de reprojection pour générer une carte de profondeur

# Obtenir les cartes de transformation :

# Une fois les matrices de rectification obtenues, cv2.initUndistortRectifyMap() est utilisée pour calculer 
# les cartes de transformation permettant de corriger la distorsion et d’appliquer la rectification.

map1x, map1y = cv2.initUndistortRectifyMap(K1, None, R1, P1, (w,h), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, None, R2, P2, (w,h), cv2.CV_32FC1)

# Appliquer la transformation sur les images :

# Enfin, cv2.remap() applique les cartes de transformation obtenues avec cv2.initUndistortRectifyMap() pour rectifier les images.

img1_rectified = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
img2_rectified = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)


# 3. Tracer les lignes épipolaires APRÈS la rectification
img1_rectified_lines = draw_epipolar_lines(img1_rectified.copy(), color=(255, 0, 0))
img2_rectified_lines = draw_epipolar_lines(img2_rectified.copy(), color=(255, 0, 0))

# Affichage des images avant et après rectification
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title("Image Gauche Originale")
plt.subplot(2, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title("Image Droite Originale")
plt.subplot(2, 2, 3)
plt.imshow(img1_rectified, cmap='gray')
plt.title("Image Gauche Rectifiée")
plt.subplot(2, 2, 4)
plt.imshow(img2_rectified, cmap='gray')
plt.title("Image Droite Rectifiée")
plt.show()