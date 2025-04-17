# matching/feature_matching.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_features_sift(image):
    """
    Detects keypoints and computes descriptors using SIFT.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    print(f"[SIFT] {len(keypoints)} keypoints détectés.")
    return keypoints, descriptors

def match_features_bf(desc1, desc2, method='sift'):
    """
    Matches feature descriptors using Brute-Force matcher and ratio test.
    """
    if desc1 is None or desc2 is None:
        print("Descriptors missing, skipping matching.")
        return [], []

    if method == 'sift':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"[{method.upper()}] {len(matches)} matches, {len(good_matches)} après ratio test.")
    return matches, good_matches

def plot_matches(img1, kp1, img2, kp2, matches, good_matches, title="Feature Matches"):
    """
    Display matches before and after ratio test.
    """
    img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_good_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_matches)
    plt.title(f"{title} (Avant ratio test)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_good_matches)
    plt.title(f"{title} (Après ratio test)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
