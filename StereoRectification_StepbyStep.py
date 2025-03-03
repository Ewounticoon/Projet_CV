import cv2
import numpy as np

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
T = np.array([baseline, 0, 0]).reshape((3, 1))

# Matrices de projection
P1 = np.hstack((K1 @ R1, np.zeros((3, 1))))  # P1 = K1 * [I | 0]
P2 = np.hstack((K2 @ R2, K2 @ T))            # P2 = K2 * [R | T]

# Fonction pour calculer le repère rectifié
def compute_rectified_coordinate_system(P1, P2):
    """
    Calcule un repère rectifié à partir des matrices de projection des caméras.
    """
    # Extraction des centres optiques
    C1 = -np.linalg.inv(P1[:, :3]) @ P1[:, 3]
    C2 = -np.linalg.inv(P2[:, :3]) @ P2[:, 3]
    
    # Calcul de la baseline
    baseline_vec = C2 - C1
    x = baseline_vec / np.linalg.norm(baseline_vec)  # Normalisation
    
    # Récupération de l'axe optique d'origine (colonne 3 de P1)
    z = P1[:, 2]
    z = z / np.linalg.norm(z)  # Normalisation
    
    # Calcul du vecteur y
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    
    # Recalcul de z pour assurer l'orthogonalité
    z = np.cross(x, y)
    
    # Matrice de rotation rectifiée
    R_rect = np.column_stack((x, y, z))
    
    return R_rect

# Fonction pour calculer les homographies
def compute_homographies(K1, K2, R1, R2, R_rect):
    """
    Calcule les matrices d'homographie H1 et H2 pour rectifier les images.
    """
    H1 = K1 @ R_rect @ R1.T @ np.linalg.inv(K1)
    H2 = K2 @ R_rect @ R2.T @ np.linalg.inv(K2)
    
    return H1, H2

# Fonction pour appliquer l'homographie sur une image
def apply_homography(image, H):
    """
    Applique la transformation d'homographie sur une image.
    """
    height, width = image.shape[:2]  # Taille de l'image
    rectified_image = cv2.warpPerspective(image, H, (width, height))  # Transformation correcte
    
    return rectified_image

# Charger les images stéréo
img1 = cv2.imread('Projet_CV/chess1/im0.png', 1)
img2 = cv2.imread('Projet_CV/chess1/im1.png', 1)

# Vérification du chargement des images
if img1 is None or img2 is None:
    print("Erreur : Impossible de charger les images.")
    exit(1)

cv2.imshow("Image 1 Originale", img1)
cv2.imshow("Image 2 Originale", img2)
cv2.waitKey(0)

# Calcul du repère rectifié
R_rect = compute_rectified_coordinate_system(P1, P2)
print("R_rect:", R_rect)

# Calcul des homographies
H1, H2 = compute_homographies(K1, K2, R1, R2, R_rect)
print("H1:", H1)
print("H2:", H2)

# Application des homographies
rectified_img1 = apply_homography(img1, H1)
rectified_img2 = apply_homography(img2, H2)

cv2.imshow("Image 1 Rectifiée", rectified_img1)
cv2.imshow("Image 2 Rectifiée", rectified_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
