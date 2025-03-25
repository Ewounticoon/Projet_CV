import numpy as np
import cv2 as cv
import glob
import os

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

objpoints = []
imgpointsL, imgpointsR = [], []

images_left = sorted(glob.glob('Projet_CV/calibration_images/left_*.jpg'))
images_right = sorted(glob.glob('Projet_CV/calibration_images/right_*.jpg'))

for fnameL, fnameR in zip(images_left, images_right):
    imgL, imgR = cv.imread(fnameL), cv.imread(fnameR)
    if imgL is None or imgR is None:
        continue
    grayL, grayR = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY), cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    retL, cornersL = cv.findChessboardCorners(grayL, (9,7), None)
    retR, cornersR = cv.findChessboardCorners(grayR, (9,7), None)

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria))
        imgpointsR.append(cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria))

if len(objpoints) > 0:
    _, K1, distL, _, _ = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    _, K2, distR, _, _ = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
    _, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, K1, distL, K2, distR, grayL.shape[::-1], criteria=criteria, flags=cv.CALIB_FIX_INTRINSIC)
    print("Matrice intrinsèque gauche:\n", K1)
    print("Distorsion gauche:\n", distL)
    print("Matrice intrinsèque droite:\n", K2)
    print("Distorsion droite:\n", distR)
    print("Rotation:\n", R)
    print("Translation:\n", T)
