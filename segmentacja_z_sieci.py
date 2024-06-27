import json
import cv2
import numpy as np
from roboflow import Roboflow
import pyrealsense2 as rs
import open3d as o3d

def predict(image_path, output_path):
    rf = Roboflow(api_key="Uc02DXLsqvZNgCT7wkWy")
    project = rf.workspace().project("segm.-instancyjna-ksd-2024")
    model = project.version("1").model

    prediction_json = model.predict(image_path, confidence=40).json()
    with open('prediction.json', 'w') as json_file:
        json.dump(prediction_json, json_file, indent=4)

    model.predict(image_path, confidence=40).save(output_path)

def create_mask(predicted_image):
    prediction_image = cv2.imread(predicted_image)
    hsv = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a mask for the blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_filled = np.zeros_like(mask)
    for cnt in contours:
        cv2.drawContours(mask_filled, [cnt], 0, (255), thickness=cv2.FILLED)

    return mask_filled

def extract(path_bag):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, path_bag)

    # Start streaming
    pipeline.start(config)

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Skipping some frames to allow for auto-exposure stabilization
    for _ in range(30):
        pipeline.wait_for_frames()

    # Get frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    pipeline.stop()

    return color_frame, depth_frame

def printing_predictions(prediction_json_path, x_values, y_values, z_values):
    with open(prediction_json_path, 'r') as file:
        data = json.load(file)

    predictions = data['predictions']

    # create dic for classes and depth values
    predicted_objects = {}

    # Iterate over each bounding box to segment the point cloud and find the point with the lowest depth
    for box in predictions:
        # Calculate bounding box corners in the depth image space from the center
        center_x, center_y = int(box['x']), int(box['y'])
        half_width, half_height = int(box['width'] / 2), int(box['height'] / 2)

        min_x, min_y = center_x - half_width, center_y - half_height
        max_x, max_y = center_x + half_width, center_y + half_height

        # Segment points within the bounding box
        segmented_points_indices = np.where(
            (x_values >= min_x) & (x_values <= max_x) &
            (y_values >= min_y) & (y_values <= max_y)
        )[0]

        if len(segmented_points_indices) == 0:
            continue  # Skip if no points are found within the bounding box

        # Find the point with the lowest depth within this segment
        lowest_depth_point_index = segmented_points_indices[np.argmin(z_values[segmented_points_indices])]

        predicted_objects[box['class']] = [z_values[lowest_depth_point_index], box['x'], box['y']]

    # segregate objects by depth ascending
    sorted_predicted_objects = sorted(predicted_objects.items(), key=lambda x: x[1][0])

    print("Detected objects in ascending order of depth:")
    for obj in sorted_predicted_objects:
        print(f'Object: {obj[0]}, centre x: {obj[1][1]}, centre y: {obj[1][2]}')


path_to_predict = 'objects.jpg'
path_predicted = 'prediction.jpg'
path_bag = 'glowna_oba_BAG.bag'

color_frame, depth_frame = extract(path_bag)

color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())
cv2.imwrite(path_to_predict, color_image)
predict(path_to_predict, path_predicted)
mask_filled = create_mask(path_predicted)

# Get the intrinsics of the depth camera
depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

# Create Open3D point cloud from RealSense depth and color frames
points = []
colors = []

for v in range(depth_intrinsics.height):
    for u in range(depth_intrinsics.width):
        z = depth_image[v, u] * depth_frame.get_units()
        if mask_filled[v, u] > 0 and z > 0:
            x = (u - depth_intrinsics.ppx) / depth_intrinsics.fx * z
            y = (v - depth_intrinsics.ppy) / depth_intrinsics.fy * z
            points.append([x, y, z])
            colors.append(color_image[v, u] / 255.0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save the point cloud to a PLY file
o3d.io.write_point_cloud('filtered_cloud_obaglowne.ply', pcd)

# Prepare data for printing_predictions
points = np.array(points)
x_values, y_values, z_values = points[:, 0], points[:, 1], points[:, 2]

# Optional: Print predictions
prediction_json_path = 'prediction.json'
printing_predictions(prediction_json_path, x_values, y_values, z_values)

# Visualize the point cloud with editing features
# This allows for axes visualization and easy rotation of the model
o3d.visualization.draw_geometries_with_editing([pcd])
