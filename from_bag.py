import pyrealsense2 as rs
import numpy as np
import cv2
import os

def align_depth_to_color(bag_file_path, output_file_path):
    # Configure the pipeline to load the bag file
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file_path)

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
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        raise RuntimeError("Could not acquire depth or color frames.")

    # Convert images to numpy arrays
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)

    # Combine images
    images = color_image
    images2 = depth_colormap
    # Save the image
    cv2.imwrite(output_file_path, images2)

    # Stop the pipeline
    pipeline.stop()

# Example usage
bag_file_path = '222.bag'  # Replace with your bag file path
output_file_path = 'to_depth.jpg'  # Replace with your desired output file path
align_depth_to_color(bag_file_path, output_file_path)