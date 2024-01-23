import json
import cv2
import numpy as np
from roboflow import Roboflow
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Initialize the Roboflow instance
rf = Roboflow(api_key="Uc02DXLsqvZNgCT7wkWy")
project = rf.workspace().project("segm.-instancyjna-ksd-2024")
model = project.version("1").model

# Infer on a local image and get the prediction in JSON format
prediction_json = model.predict("to_network.jpg", confidence=40).json()

# Print the prediction JSON
print(prediction_json)

# Save the prediction JSON to a file
with open('prediction.json', 'w') as json_file:
    json.dump(prediction_json, json_file, indent=4)

# Visualize your prediction
model.predict("to_network.jpg", confidence=40).save("prediction.jpg")

# Load the prediction image and the other photo
prediction_image = cv2.imread('prediction.jpg')
other_photo = cv2.imread('to_network.jpg')  # Replace with your photo's path

# Ensure other photo is the same size as prediction image
other_photo = cv2.resize(other_photo, (prediction_image.shape[1], prediction_image.shape[0]))

# Convert the prediction image to the HSV color space
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

# Convert the mask to an alpha channel where mask is 0, alpha is 255 (opaque)
alpha_channel = np.where(mask_filled != 0, 255, 0).astype('uint8')

# Convert other photo to BGRA and add the alpha channel
other_photo_bgra = cv2.cvtColor(other_photo, cv2.COLOR_BGR2BGRA)
other_photo_bgra[:, :, 3] = alpha_channel

# Save the modified photo with transparency applied
cv2.imwrite('other_photo_transparent.png', other_photo_bgra)

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, '222.bag')

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
aligned_depth_frame = aligned_frames.get_depth_frame()

# Convert images to numpy arrays
depth_image = np.asanyarray(aligned_depth_frame.get_data())
masked_depth = np.where(mask_filled != 0, depth_image, np.nan)
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
pipeline.stop()
depth_width = aligned_depth_frame.get_width()
depth_height = aligned_depth_frame.get_height()
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

# Prepare for 3D scatter plot
xv, yv = np.meshgrid(range(depth_width), range(depth_height), indexing='xy')

# Flatten the arrays for plotting
x_values, y_values, z_values = xv.flatten(), yv.flatten(), normalized_masked_depth.flatten()
valid_points = ~np.isnan(z_values)  # Identify valid points where z_values are not NaN
x_values, y_values, z_values = x_values[valid_points], y_values[valid_points], z_values[valid_points]

# Filter the color data
colors = resized_color_image.reshape(-1, 3) / 255
colors = colors[valid_points]  # Only keep colors for valid points

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_values, y_values, z_values, c=colors, marker='.', s=1)  # Use z_values for the Z-axis

ax.set_xlabel('Width')
ax.set_ylabel('Height')
ax.set_zlabel('Normalized Depth')
ax.set_title('3D Scatter Plot of Masked Depth Data with RGB Colors')

plt.show()
