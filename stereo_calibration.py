import numpy as np
import cv2 as cv
import glob

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

    rows = 6  # number of checkerboard rows.
    columns = 9  # number of checkerboard columns.
    world_scaling = 65  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

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
    return ret, mtx, dist,

ret, mtx, dist = calibrate_camera(path,fx,cx,fy,cy)
ret2, mtx2, dist2 = calibrate_camera(path1,fx3,cx3,fy3,cy3)

images_good = 'images_good/*.png'
images_good1 = 'images_good2/*.png'
def stereo_calibration(mtx1,dist1,mtx2,dist2,images_good,images_good1):
    c1_images_names = sorted(glob.glob(images_good))
    c2_images_names = sorted(glob.glob(images_good1))

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 6  # number of checkerboard rows.
    columns = 9  # number of checkerboard columns.
    world_scaling = 65  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (6, 9), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (6, 9), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (6, 9), corners1, c_ret1)
            cv.imshow('img', frame1)

            cv.drawChessboardCorners(frame2, (6, 9), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,
                                                                 dist1,
                                                                 mtx2, dist2, (width, height), criteria=criteria,
                                                                 flags=stereocalibration_flags)

    print(ret)
    return R, T

R, T = stereo_calibration(mtx, dist, mtx2, dist2, images_good, images_good1)





