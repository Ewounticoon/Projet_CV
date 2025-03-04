import cv2
import numpy as np
import matplotlib.pyplot as plt


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

# Étape 1 : Définition du système de coordonnées rectifié

def rectified_coord_system():
    # Axe X : Direction normalisée de la baseline
    x_axis = T / np.linalg.norm(T)

    # Axe Z : Même direction que l'axe optique initial
    z_axis = np.array([0, 0, 1])

    # Axe Y : Produit vectoriel pour garantir l'orthogonalité
    y_axis = np.cross(z_axis, x_axis)

    # Construction de la nouvelle matrice de rotation rectifiée
    R_rect = np.column_stack((x_axis, y_axis, z_axis)) # Cette méthode permet de stack3 vecteurs pour former une matrice 3x3 avec les 3 vecteurs [ X Y Z ] 
    return R_rect

# Étape 2 : Calcul des homographies

def compute_homographie(R_rect):
    H1 = K1 @ R_rect @ R1.T @ np.linalg.inv(K1)
    H2 = K2 @ R_rect @ R2.T @ np.linalg.inv(K2)

    return H1,H2


# Fonction pour appliquer la transformation homographique
def warp_image(image, H, size):
    return cv2.warpPerspective(image, H, size)
 
# Cette méthode prend l'image qui est une grande matrice
# Chaque case de la matrice est un pixel aux coordonnées x et y
# On applique I' = H*I avec I les coordonnées homogène du pixel dans l'image de base et I' les coordonnées homogène du pixel dans l'image rectifié
# On attribue la valeur du pixel de I au nouvelles coordonnées I'
# On obtient une nouvelle image et rectifié

def main():
    # Charger les images stéréo
    img1 = cv2.imread('Projet_CV/chess1/im0.png', 1)
    img2 = cv2.imread('Projet_CV/chess1/im1.png', 1)


    # Vérifier si les images ont été chargées correctement
    if img1 is None or img2 is None:
        print("Erreur : Impossible de charger les images.")
        return

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

if __name__ == "__main__":
    main()