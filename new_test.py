import numpy as np
import cv2 as cv
import glob
import os
 
# Critères d'arrêt pour l'affinement des coins
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Préparation des points 3D pour un échiquier 9x7
objp = np.zeros((9 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)
 
# Listes pour stocker les points
objpoints = []  # Points 3D dans le monde réel
imgpoints = []  # Points 2D détectés dans l'image
 
# Chargement des images
images = glob.glob('calibration_images_files/*.jpg')
print(f"{len(images)} images trouvées.")
 
# Dossier pour sauvegarder les résultats
save_path = "output_images"
os.makedirs(save_path, exist_ok=True)
 
for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Erreur : Impossible de lire l'image {fname}")
        continue
 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Détection des coins de l'échiquier (9x7)
    ret, corners = cv.findChessboardCorners(gray, (9,7), None)
 
    if ret and corners.shape[0] == objp.shape[0]:  # Vérifie qu'on a bien 63 points
 
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Dessiner les coins détectés
        cv.drawChessboardCorners(img, (9,7), corners2, ret)
 
        # Sauvegarder l'image annotée
        filename = os.path.basename(fname)
        cv.imwrite(os.path.join(save_path, f"{filename}"), img)
 
        cv.imshow('img', img)
        cv.waitKey(500)
 
cv.destroyAllWindows()
 
# Vérifier qu'on a bien collecté des points avant la calibration
if len(objpoints) == len(imgpoints) and len(objpoints) > 0:
    print("Calibration en cours...")
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 
    print(f"K1 (Matrice intrinsèque) : \n{mtx}\n")
    print(f"D1 (Distorsion) : \n{dist}\n")
else:
    print(" Erreur : pas assez de points pour calibrer la caméra.")