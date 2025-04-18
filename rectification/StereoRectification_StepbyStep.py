import cv2
import numpy as np
import matplotlib.pyplot as plt

class Rectif_Stereo():

   def __init__(self, K1, K2, R1, R2, T):
      self.K1=K1
      self.K2=K2
      self.R1=R1
      self.R2=R2
      self.T=T

# Étape 1 : Définition du système de coordonnées rectifié

   def rectified_coord_system(self):
      # Axe X : Direction normalisée de la baseline
      x_axis = self.T / np.linalg.norm(self.T)

      # Axe Z : Même direction que l'axe optique initial
      z_axis = np.array([0, 0, 1])
   
      # Axe Y : Produit vectoriel pour garantir l'orthogonalité
      y_axis = np.cross(z_axis, x_axis)

      # Construction de la nouvelle matrice de rotation rectifiée
      R_rect = np.column_stack((x_axis, y_axis, z_axis)) # Cette méthode permet de stack3 vecteurs pour former une matrice 3x3 avec les 3 vecteurs [ X Y Z ] 
      return R_rect

    # Étape 2 : Calcul des homographies

   def compute_homographie(self,R_rect):
      H1 = self.K1 @ R_rect @ self.R1.T @ np.linalg.inv(self.K1)
      H2 = self.K2 @ R_rect @ self.R2.T @ np.linalg.inv(self.K2)

      return H1,H2

   def draw_epipolar_lines(self,img_left, img_right, num_lines=10):
      """
      Trace les lignes épipolaires horizontales sur les deux images rectifiées.

      :param img_left: Image gauche rectifiée (numpy array)
      :param img_right: Image droite rectifiée (numpy array)
      :param num_lines: Nombre de lignes épipolaires à tracer
      :return: Images avec lignes épipolaires dessinées
      """
      # Copie pour dessin
      img_left_lines = img_left.copy()
      img_right_lines = img_right.copy()

      height, width = img_left.shape[:2]
      step = height // num_lines

      for i in range(step, height, step):
         color = tuple(np.random.randint(0, 255, 3).tolist())
         cv2.line(img_left_lines, (0, i), (width, i), color, 1)
         cv2.line(img_right_lines, (0, i), (width, i), color, 1)

      cv2.imshow("Image_left",img_left_lines)
      cv2.imshow("Image_right",img_right_lines)
      #cv2.imwrite("img_left_lines.png",img_left_lines)
      #cv2.imwrite("img_right_lines.png",img_right_lines)

      cv2.waitKey(0)


    # Fonction pour appliquer la transformation homographique
   def warp_image(self,image, H, size):
      return cv2.warpPerspective(image, H, size)
    
   # Cette méthode prend l'image qui est une grande matrice
   # Chaque case de la matrice est un pixel aux coordonnées x et y
   # On applique I' = H*I avec I les coordonnées homogène du pixel dans l'image de base et I' les coordonnées homogène du pixel dans l'image rectifié
   # On attribue la valeur du pixel de I au nouvelles coordonnées I'
   # On obtient une nouvelle image et rectifié


   def stereo_rectify(self,K0, K1, R, T, image_size):
      """
      Perform stereo rectification.
      :param K0: Intrinsic matrix of left camera
      :param K1: Intrinsic matrix of right camera
      :param R: Rotation matrix between the cameras
      :param T: Translation vector between the cameras
      :param image_size: Tuple (width, height)
      :return: Rectification maps
      """
      # Définir des coefficients de distorsion nuls (si aucune distorsion n'est à corriger)
      # On crée des tableaux de forme (1,5) pour respecter le format attendu par OpenCV.
      #distCoeffs0 = np.zeros((1, 5), dtype=np.float64)
      #distCoeffs1 = np.zeros((1, 5), dtype=np.float64)
      distCoeffs1= np.array([3.38366672e-02, -1.20838533e-01, 1.17279578e-04, 2.41881867e-03, 9.83446269e-02])
      distCoeffs0= np.array([0.1801551, -0.39853009, -0.00040739, 0.00367763, 0.01424179])
      # Attention à l'ordre des paramètres : 
      # (cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, ...)
      R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
         K0, distCoeffs0, K1, distCoeffs1, image_size, R, T,
         flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
      )
      return R1, R2, P1, P2, Q

def main():
   pass
   ## Charger les images stéréo
   #img1 = cv2.imread('chess1/im0.png', 1)
   #img2 = cv2.imread('chess1/im1.png', 1)
   ## Vecteur translation (baseline le long de l'axe x)
   #baseline=111.53
   #    # Matrices intrinsèques
   #K1 = np.array([[1758.23, 0, 953.34],
   #                [0, 1758.23, 552.29],
   #                [0, 0, 1]])
#
   #K2 = np.array([[1758.23, 0, 953.34],
   #                [0, 1758.23, 552.29],
   #                [0, 0, 1]])
#
   ## Rotation : identique pour les deux (parallèles)
   ## Rotation identité pour les deux caméras (elles sont déjà alignées)
   #R1 = np.eye(3)
   #R2 = np.eye(3)
#
   ## Translation : baseline de 111.53 mm sur l’axe X
   #T = np.array([111.53, 0, 0])  # En millimètres
#
   ##T = np.array([-15.27075501, -25.26244818, 102.40720507])
   ##print(T)
#
#
   #rectif=Rectif_Stereo(K1,K2,R1,R2,T)
   #T = np.array([baseline, 0, 0])
   ## Vérifier si les images ont été chargées correctement
   #if img1 is None or img2 is None:
   #    print("Erreur : Impossible de charger les images.")
   #    return
   ## Convertir en RGB pour affichage correct avec matplotlib
   #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
   #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
   ## Dimensions de l'image
   #h, w, _ = img1.shape
   #R_rect=rectif.rectified_coord_system()
   #H1,H2 = rectif.compute_homographie(R_rect)
   ## Application des homographies
   #img1_rectified = rectif.warp_image(img1, H1, (w, h))
   #img2_rectified = rectif.warp_image(img2, H2, (w, h))
   ## Affichage des images avant et après rectification
   #plt.figure(figsize=(10, 5))
   #plt.subplot(2, 2, 1)
   #plt.imshow(img1, cmap='gray')
   #plt.title("Image Gauche Originale")
   #plt.subplot(2, 2, 2)
   #plt.imshow(img2, cmap='gray')
   #plt.title("Image Droite Originale")
   #plt.subplot(2, 2, 3)
   #plt.imshow(img1_rectified, cmap='gray')
   #plt.title("Image Gauche Rectifiée")
   #plt.subplot(2, 2, 4)
   #plt.imshow(img2_rectified, cmap='gray')
   #plt.title("Image Droite Rectifiée")
   #plt.show()

if __name__ == "__main__":
    main()