import cv2
import numpy as np
import glob
import os

# Critères pour l'affinage des coins
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.001)

# Création des points 3D (grille d'échiquier)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Dossiers contenant les images de calibration
left_images = glob.glob('Projet_CV/calibration_images/*.jpg')
right_images = glob.glob('Projet_CV/calibration_images/*.jpg')

print(f"Nombre d'images chargées pour la caméra gauche : {len(left_images)}")
print(f"Nombre d'images chargées pour la caméra droite : {len(right_images)}")

# Vérifie si le chargement des images a fonctionné
if len(left_images) == 0 or len(right_images) == 0:
    raise ValueError(" Aucune image trouvée ! Vérifie les chemins et extensions.")

objpoints = []  # Points 3D du monde réel
imgpoints_left = []  # Points 2D pour la caméra gauche
imgpoints_right = []  # Points 2D pour la caméra droite

# Lire les images et détecter les coins
for fname_left, fname_right in zip(left_images, right_images):
    img_left = cv2.imread(fname_left)
    img_right = cv2.imread(fname_right)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (9, 6), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (9, 6), None)

    if ret_left and ret_right:
        objpoints.append(objp)

        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(corners2_left)
        imgpoints_right.append(corners2_right)

# Calibration individuelle des caméras
ret_left, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_right, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

print(f"K1 (Caméra gauche) : \n{K1}\n")
print(f"D1 (Distorsion gauche) : \n{D1}\n")
print(f"K2 (Caméra droite) : \n{K2}\n")
print(f"D2 (Distorsion droite) : \n{D2}\n")

# Calibration stéréo
flags = cv2.CALIB_FIX_INTRINSIC  # On suppose que les paramètres intrinsèques sont fixes
ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, K1, D1, K2, D2, gray_left.shape[::-1],
    criteria=criteria, flags=flags
)

print(f"Rotation entre les caméras (R) : \n{R}\n")
print(f"Translation entre les caméras (T) : \n{T}\n")
print(f"Matrice essentielle (E) : \n{E}\n")
print(f"Matrice fondamentale (F) : \n{F}\n")
