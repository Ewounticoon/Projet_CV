import cv2
import numpy as np
import glob

def calibrate_camera(img_dir='calibration_images_files/'):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

    objpoints = []
    imgpointsL, imgpointsR = [], []

    images_left = sorted(glob.glob(img_dir + 'left_*.jpg'))
    images_right = sorted(glob.glob(img_dir + 'right_*.jpg'))

    for fnameL, fnameR in zip(images_left, images_right):
        imgL, imgR = cv2.imread(fnameL), cv2.imread(fnameR)
        if imgL is None or imgR is None:
            continue
        grayL, grayR = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, (9,7), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (9,7), None)

        if retL and retR:
            objpoints.append(objp)
            imgpointsL.append(cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria))
            imgpointsR.append(cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria))

    if len(objpoints) > 0:
        _, K1, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
        _, K2, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
        _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objpoints, imgpointsL, imgpointsR, K1, distL, K2, distR, grayL.shape[::-1],
            criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
        print("Matrice intrinsèque gauche:\n", K1)
        print("Distorsion gauche:\n", distL)
        print("Matrice intrinsèque droite:\n", K2)
        print("Distorsion droite:\n", distR)
        print("Rotation:\n", R)
        print("Translation:\n", T)
        return K1, K2, R, T
    return None, None, None, None
