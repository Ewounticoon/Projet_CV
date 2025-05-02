# Projet computer Vision : Reconstruction 3D par Stéréo Vision

Ce projet de **vision par ordinateur** a été réalisé dans le cadre d’un cours à **Polytech Dijon**, proposé par **Y. Fougerolle**.  
L’objectif est de **comprendre le principe de la stéréovision** en expérimentant la reconstruction 3D à partir de paires d’images.

## 🔍 Objectifs

- Comprendre la **géométrie épipolaire** (calibration, rectification)
- Appliquer la **reconstruction éparse** à partir de points clés
- Mettre en œuvre une **reconstruction dense** (disparité + reprojection 3D)
- Visualiser un **nuage de points 3D** à partir de deux caméras virtuelles

## 🧪 Ce qu’on a fait

### Reconstruction éparse (Sparse)
- Détection de points clés (SIFT)
- Appariement avec FLANN + filtrage RANSAC
- Estimation de la matrice essentielle
- Triangulation de points 3D

### Reconstruction dense
- Calibration et rectification stéréo
- Calcul de la **carte de disparité** (StereoBM / StereoSGBM)
- Reprojection en 3D avec la matrice Q
- Affichage et export du nuage de points

> 🔧 On s’est concentré principalement sur la reconstruction **dense**, plus réaliste pour un nuage complet.

## 🛠️ Technologies

- Python 3
- OpenCV (cv2)
- Numpy / Matplotlib

---

