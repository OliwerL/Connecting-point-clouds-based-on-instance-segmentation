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

# Load the prediction image and the other photo
prediction_image = cv2.imread('prediction.jpg')
other_photo = cv2.imread('to_depth.jpg')  # Replace with your photo's path

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

# Draw the contours and fill them in on the mask
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
