import numpy as np
import cv2
import glob
import pickle
import json
import os


path = 'images/*.png'
useful_path = 'firstcamera/*.png'
chessboardSize = (9, 6)
frameSize = (640, 480)
fx = 0
fy = 0
cx = 0
cy = 0
initial_camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype='float32')
initial_dist_coeffs = np.zeros((5, 1), dtype='float32')  # Dostosuj do odpowiedniej długości

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1, 2)

objpoints = []# 3d points
imgpoints = []# 2d points

size_ches_square = 65
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

ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], initial_camera_matrix, initial_dist_coeffs,
    flags=cv2.CALIB_USE_INTRINSIC_GUESS
)
calibration_data = {
    "camera_matrix": mtx,
    "distortion_coefficients": dist,
    "rotation_vectors": rvecs,
    "translation_vectors": tvecs,
    "stdDeviationsIntrinsics": stdDeviationsIntrinsics,
    "stdDeviationsExtrinsics": stdDeviationsExtrinsics,
    "perViewErrors": perViewErrors
}

with open("camera_calibration_data.pkl", "wb") as file:
    pickle.dump(calibration_data, file)

calibration_data_json = {
    "ret": ret,
    "camera_matrix": mtx.tolist(),
    "distortion_coefficients": dist.tolist(),
    "rotation_vectors": [rvec.tolist() for rvec in rvecs],
    "translation_vectors": [tvec.tolist() for tvec in tvecs],
    "stdDeviationsIntrinsics": stdDeviationsIntrinsics.tolist(),
    "stdDeviationsExtrinsics": stdDeviationsExtrinsics.tolist(),
    "perViewErrors": perViewErrors.tolist()
}

# Zapisz dane kalibracyjne do pliku JSON
with open("camera_calibration_data.json", "w") as json_file:
    json.dump(calibration_data_json, json_file, indent=4)

print("Dane kalibracyjne zostały zapisane w formacie JSON.")


for iname in images:
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
    corrected_image_path = os.path.join(useful_path, 'corrected_' + filename)
    cv2.imwrite(corrected_image_path, dst)
