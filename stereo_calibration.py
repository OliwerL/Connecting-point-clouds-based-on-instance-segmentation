import numpy as np
import cv2
import glob
import pickle

with open('camera1_calibration_data.pkl', 'rb') as f:
    mtx1, dist1 = pickle.load(f)

with open('camera2_calibration_data.pkl', 'rb') as f:
    mtx2, dist2 = pickle.load(f)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessboardSize = (9,6)
objp = np.zeros((np.prod(chessboardSize), 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

objpoints = []
imgpoints1 = []
imgpoints2 = []

images1 = glob.glob('path_to_left_camera_images/*.png')
images2 = glob.glob('path_to_right_camera_images/*.png')

for img1, img2 in zip(images1, images2):
    imgL = cv2.imread(img1)
    imgR = cv2.imread(img2)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Znajdź narożniki na planszy
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

    if retL and retR:
        objpoints.append(objp)

        corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpoints1.append(corners2L)

        corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpoints2.append(corners2R)

# Stereokalibracja
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2,
    mtx1, dist1,
    mtx2, dist2,
    grayL.shape[::-1],
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)


print("Macierz rotacji między kamerami:\n", R)
print("Wektor translacji między kamerami:\n", T)

# Zapisz wyniki stereokalibracji do pliku za pomocą pickle
stereo_calibration_params = {
    "rotation_matrix": R,
    "translation_vector": T
}

with open('stereo_calibration_params.pkl', 'wb') as f:
    pickle.dump(stereo_calibration_params, f)

print("Dane stereokalibracji zostały zapisane.")