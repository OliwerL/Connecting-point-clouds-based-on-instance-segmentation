import numpy as np
import cv2 as cv
import glob
import json

path = 'images/*.png'
path1 = 'images3/*.png'
chessboardSize = (9, 6)
frameSize = (640, 480)
fx1 = 607.27783203125
fy1 = 607.0103149414062
cx1 = 319.4129333496094
cy1 = 253.3705291748047
fx = 607.13525390625
fy = 606.7884521484375
cx = 316.97607421875
cy = 250.20372009277344
fx3 = 608.4392700195312
fy3 = 608.0927734375
cx3 = 324.9110412597656
cy3 = 250.14309692382812

def calibrate_camera(images_folder, fx,cx,fy,cy):
    initial_camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype='float32')
    initial_dist_coeffs = np.zeros((5, 1), dtype='float32')  # Dostosuj do odpowiedniej długości
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 5  # number of checkerboard rows.
    columns = 8  # number of checkerboard columns.
    world_scaling = 1.  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv.calibrateCameraExtended(
        objpoints, imgpoints, gray.shape[::-1], initial_camera_matrix, initial_dist_coeffs,
        flags=cv.CALIB_USE_INTRINSIC_GUESS
    )
    return ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors

ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = calibrate_camera(path,fx,cx,fy,cy)
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

ret2, mtx2, dist2, rvecs2, tvecs2, stdDeviationsIntrinsics2, stdDeviationsExtrinsics2, perViewErrors2 = calibrate_camera(path1,fx3,cx3,fy3,cy3)
calibration_data_json = {
    "ret": ret2,
    "camera_matrix": mtx2.tolist(),
    "distortion_coefficients": dist2.tolist(),
    "rotation_vectors": [rvec.tolist() for rvec2 in rvecs],
    "translation_vectors": [tvec.tolist() for tvec2 in tvecs],
    "stdDeviationsIntrinsics": stdDeviationsIntrinsics2.tolist(),
    "stdDeviationsExtrinsics": stdDeviationsExtrinsics.tolist(),
    "perViewErrors": perViewErrors.tolist()
}

# Zapisz dane kalibracyjne do pliku JSON
with open("camera_calibration_data3.json", "w") as json_file:
    json.dump(calibration_data_json, json_file, indent=4)


