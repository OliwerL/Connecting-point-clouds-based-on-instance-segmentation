import numpy as np
import cv2
import glob
import pickle
import json
import os

path = 'cameraCalibration/images/*.png'
useful_path = '/images/useful/firstcamera/*.png'
chessboardSize = (9, 6)
frameSize = (640, 480)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1, 2)

objpoints = []# 3d points
imgpoints = []# 2d points

size_ches_square = 20
objp = objp * size_ches_square

images = glob.glob(path)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Rysowanie i wyświetlanie narożników
        img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
calibration_data = {
    "camera_matrix": mtx,
    "distortion_coefficients": dist,
    "rotation_vectors": rvecs,
    "translation_vectors": tvecs
}

with open("camera_calibration_data.pkl", "wb") as file:
    pickle.dump(calibration_data, file)

calibration_data_json = {
    "camera_matrix": mtx.tolist(),
    "distortion_coefficients": dist.tolist(),
    "rotation_vectors": [rvec.tolist() for rvec in rvecs],
    "translation_vectors": [tvec.tolist() for tvec in tvecs]
}

# Zapisz dane kalibracyjne do pliku JSON
with open("camera_calibration_data.json", "w") as json_file:
    json.dump(calibration_data_json, json_file, indent=4)

print("Dane kalibracyjne zostały zapisane w formacie JSON.")

images2 = glob.glob(useful_path)
for iname in images2:
    img = cv2.imread(iname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Korekcja zniekształceń
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Przycinanie obrazu
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Zapisywanie skorygowanego obrazu
    filename = os.path.basename(iname)
    corrected_image_path = os.path.join('path/to/corrected_images', 'corrected_' + filename)
    cv2.imwrite(corrected_image_path, dst)
