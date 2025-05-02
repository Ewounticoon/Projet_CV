import cv2
import numpy as np
import glob
import os

# Critères pour l'affinage des coins
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.001)

# Création des points 3D (grille d'échiquier)
objp = np.zeros((7*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

# Dossiers contenant les images de calibration
images_droites = glob.glob('calibration_images_2/droite_*.jpg')
images_gauches = glob.glob('calibration_images_2/gauche_*.jpg')

print(f"Nombre d'images chargées pour la caméra gauche : {len(images_gauches)}")
print(f"Nombre d'images chargées pour la caméra droite : {len(images_droites)}")

objpoints = []  # Points 3D du monde réel
imgpoints_gauche = []  # Points 2D pour la caméra gauche
imgpoints_droite = []  # Points 2D pour la caméra droite

# Lire les images et détecter les coins
for fname_gauche, fname_droite in zip(images_gauches, images_droites):
    img_gauche = cv2.imread(fname_gauche)
    img_droite = cv2.imread(fname_droite)

    niveau2gris_gauche = cv2.cvtColor(img_gauche, cv2.COLOR_BGR2GRAY)
    niveau2gris_droite = cv2.cvtColor(img_droite, cv2.COLOR_BGR2GRAY)

    ret_gauche, coins_gauche = cv2.findChessboardCorners(niveau2gris_gauche, (9, 7), None)
    ret_droite, coins_droite = cv2.findChessboardCorners(niveau2gris_droite, (9, 7), None)

    if ret_gauche and ret_droite:
        objpoints.append(objp)

        coins2_gauche = cv2.cornerSubPix(niveau2gris_gauche, coins_gauche, (11, 11), (-1, -1), criteria)
        coins2_droite = cv2.cornerSubPix(niveau2gris_droite, coins_droite, (11, 11), (-1, -1), criteria)

        imgpoints_gauche.append(coins2_gauche)
        imgpoints_droite.append(coins2_droite)

# Calibration individuelle
ret_g, K1, D1, rvecs_g, tvecs_g = cv2.calibrateCamera(objpoints, imgpoints_gauche, niveau2gris_gauche.shape[::-1], None, None)
ret_d, K2, D2, rvecs_d, tvecs_d = cv2.calibrateCamera(objpoints, imgpoints_droite, niveau2gris_droite.shape[::-1], None, None)

print(f"\nK1 (Gauche) : \n{K1}")
print(f"D1 (Distorsion gauche) : \n{D1}")
print(f"\nK2 (Droite) : \n{K2}")
print(f"D2 (Distorsion droite) : \n{D2}")


# Correction de la distortion pour l'image de gauche
map1_g, map2_g = cv2.initUndistortRectifyMap(K1, D1, None, K1, niveau2gris_gauche.shape[::-1], cv2.CV_16SC2)
undistorted_gauche = cv2.remap(niveau2gris_gauche, map1_g, map2_g, interpolation=cv2.INTER_LINEAR)

# Correction de la distortion pour l'image de droite
map1_d, map2_d = cv2.initUndistortRectifyMap(K2, D2, None, K2, niveau2gris_droite.shape[::-1], cv2.CV_16SC2)
undistorted_droite = cv2.remap(niveau2gris_droite, map1_d, map2_d, interpolation=cv2.INTER_LINEAR)


# Calibration stéréo
flags = cv2.CALIB_FIX_INTRINSIC
retS, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_gauche, imgpoints_droite,
    K1, D1, K2, D2, niveau2gris_gauche.shape[::-1],
    criteria=criteria, flags=flags
)

print(f"\nRotation entre les caméras (R) :\n{R}")
print(f"Translation entre les caméras (T) :\n{T}")
print(f"Matrice essentielle (E) :\n{E}")
print(f"Matrice fondamentale (F) :\n{F}")

