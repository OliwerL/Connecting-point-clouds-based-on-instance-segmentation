import json
import cv2
import numpy as np
from roboflow import Roboflow

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

image = cv2.imread('prediction.jpg')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of blue color in HSV
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Create a mask for the blue color
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours and fill them in on the mask, not the original image
mask_filled = np.zeros_like(mask)
for cnt in contours:
    cv2.drawContours(mask_filled, [cnt], 0, (255), thickness=cv2.FILLED)

# Convert the mask to a 3-channel image
mask_colored = cv2.merge([mask_filled, mask_filled, mask_filled])

# Create an alpha channel from the mask
alpha_channel = np.where(mask_filled==255, 255, 0).astype('uint8')

# Add the alpha channel to the original image
image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
image_bgra[:, :, 3] = alpha_channel

# Save the modified image with transparent background
cv2.imwrite('objects_transparent_background.png', image_bgra)