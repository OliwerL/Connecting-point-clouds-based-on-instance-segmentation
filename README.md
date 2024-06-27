# Connecting point clouds based on instance segmentation

## Introduction

The aim of this project was to connect point clouds from multiple stereovision cameras and create a 3d model of an item in the scene viewed from different angles. The items were cut from the scene after perfoming instance segmentation on the images and then point clouds were merged and cleaned.

## How it works

### 1. Calibrating the cameras

Three RealSense d435i cameras (rotated roughly by 90 degrees and located 1.6 meters away from each other) were used. Individual cameras have been calibrated using RealSense viewer desktop app. After getting .bag and specification files the cameras have been stereo calibrated in order to receive transformation and rotation matrixes showing the differences between their coordinate systems.

```python
mtx1 = np.array([[intrinsics1['fx'], 0, intrinsics1['cx']],
                     [0, intrinsics1['fy'], intrinsics1['cy']],
                     [0, 0, 1]])
    dist1 = np.array(intrinsics1['coeffs'])

    mtx2 = np.array([[intrinsics2['fx'], 0, intrinsics2['cx']],
                     [0, intrinsics2['fy'], intrinsics2['cy']],
                     [0, 0, 1]])
    dist2 = np.array(intrinsics2['coeffs'])

    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        mtx1, dist1, mtx2, dist2, gray1.shape[::-1],
        criteria=criteria)
```

  <img src="https://github.com/Ramosa5/Connecting-point-clouds-based-on-instance-segmentation/assets/108287744/ed9f8386-1690-43ea-8570-18d24dc22523" alt="Calibration" height="250">

### 2. Cutting items from the scene

Having calibrated the cameras retrieval of the objects from the scene is needed to get a clear 3d model. Firstly point clouds and photos are extracted from .bag files. Objects of interest are detected using a YOLOv8 neural network which has been tought to recognize different objects put in our scene. Then a mask is created to highlight the points matched with the items. The final outcome is a convolution of the mask and and point cloud.

<img src="https://github.com/Ramosa5/Connecting-point-clouds-based-on-instance-segmentation/assets/108287744/8dd7abbe-ef24-4f46-b269-5c8b7a2a0168" alt="Cut cloud" height="250">

### 3. Connecting clouds

After getting 3 point clouds they are merged in order to form a 3d model of the items. Cloud from the middle camera acts as the main cloud and the other two are transformed and rotated using previously calculated matrixes to match the main's coordinate systems.

  <img src="https://github.com/Ramosa5/Connecting-point-clouds-based-on-instance-segmentation/assets/108287744/58f7e0a0-9d9e-4fef-9640-3018a46ed2a0" alt="Merged cloud" height="250">

### 4. Smoothing and clearing the final cloud

The outcome cloud from the previous step has various noises so a smoothing and clearing operation has to be performed. That is achieved by finding outliers from the cloud and removing them.

```python
    def remove_isolated_points(point_cloud, nb_neighbors, std_ratio):
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = point_cloud.select_by_index(ind)
    return inlier_cloud
```

<p align="center">
  <img src="https://github.com/Ramosa5/Connecting-point-clouds-based-on-instance-segmentation/assets/108287744/43788c92-b574-4158-b5bf-5244c658bae3" height="200">
  <img src="https://github.com/Ramosa5/Connecting-point-clouds-based-on-instance-segmentation/assets/108287744/0d98620c-3468-4f60-a304-34b72c7c45b2" height="200">
</p>

