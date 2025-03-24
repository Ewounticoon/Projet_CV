import cv2
import matplotlib.pyplot as plt

# 1. Lire les images rectifiées
imgL=None
imgR=None

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
