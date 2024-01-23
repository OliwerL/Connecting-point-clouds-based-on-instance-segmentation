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

    # prediction_json = model.predict(image_path, confidence=40).json()
    # print(prediction_json)
    # with open('prediction.json', 'w') as json_file:
    #     json.dump(prediction_json, json_file, indent=4)

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


path_to_predict = 'objects.jpg'
path_predicted = 'prediction.jpg'
path_bag = '222.bag'

color_frame, depth_frame = extract(path_bag)

color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())
cv2.imwrite(path_to_predict, color_image)
predict(path_to_predict, path_predicted)
mask_filled = create_mask(path_predicted)


masked_depth = np.where(mask_filled != 0, depth_image, np.nan)
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
depth_width = depth_frame.get_width()
depth_height = depth_frame.get_height()
# Create a meshgrid using the extracted width and height

normalized_depth = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
min_depth = np.nanmin(masked_depth)
max_depth = np.nanmax(masked_depth)
normalized_masked_depth = (masked_depth - min_depth) / (max_depth - min_depth)
normalized_masked_depth = np.nan_to_num(normalized_masked_depth, nan=0)

# Convert to 8-bit format
normalized_masked_depth_8bit = (normalized_masked_depth * 255).astype(np.uint8)

# Apply colormap
colored_masked_depth = cv2.applyColorMap(normalized_masked_depth_8bit, cv2.COLORMAP_JET)

# Resize color image to match depth image's resolution if they are different
color_height, color_width = depth_colormap.shape[:2]
if (color_width, color_height) != (depth_width, depth_height):
    resized_color_image = cv2.resize(depth_colormap, (depth_width, depth_height), interpolation=cv2.INTER_AREA)
else:
    resized_color_image = depth_colormap

# # Prepare for 3D scatter plot
xv, yv = np.meshgrid(range(depth_width), range(depth_height), indexing='xy')

depth_threshold = 100
depth_threshold2 = 1550
mask_filled_flat = mask_filled.flatten()

# Flatten the arrays for plotting
x_values, y_values, z_values = xv.flatten(), yv.flatten(), masked_depth.flatten()

# Identify valid points where z_values are not NaN, corresponding mask_filled values are zero, and depth is below the threshold
valid_points = (~np.isnan(z_values)) & (z_values > depth_threshold) & (z_values < depth_threshold2)
x_values, y_values, z_values = x_values[valid_points], y_values[valid_points], z_values[valid_points]

# Filter the color data
colors = resized_color_image.reshape(-1, 3)
colors = colors[valid_points]  # Only keep colors for valid points

# Create Open3D point cloud with only valid points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack((x_values, y_values, z_values)).T)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors

# Visualize the point cloud with editing features
# This allows for axes visualization and easy rotation of the model
o3d.visualization.draw_geometries_with_editing([pcd])