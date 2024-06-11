import pyrealsense2 as rs
import numpy as np
import open3d as o3d


def extract_point_cloud_from_bag(path_bag):
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

    # Stop pipeline
    pipeline.stop()

    return color_frame, depth_frame


def convert_to_open3d_point_cloud(color_frame, depth_frame):
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert depth image to point cloud
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vtx)

    # Extract color
    rgb = color_image.reshape(-1, 3) / 255.0  # Normalize to [0,1]
    point_cloud.colors = o3d.utility.Vector3dVector(rgb)

    return point_cloud


def save_point_cloud_to_ply(point_cloud, ply_filename):
    o3d.io.write_point_cloud(ply_filename, point_cloud)


# Paths to bag files and output PLY files
bag_file1 = 'glowna1BAG.bag'
bag_file2 = 'prawa1BAG.bag'
ply_file1 = 'glowny1.ply'
ply_file2 = 'prawa1.ply'

# Extract and convert point clouds
color_frame1, depth_frame1 = extract_point_cloud_from_bag(bag_file1)
color_frame2, depth_frame2 = extract_point_cloud_from_bag(bag_file2)

point_cloud1 = convert_to_open3d_point_cloud(color_frame1, depth_frame1)
point_cloud2 = convert_to_open3d_point_cloud(color_frame2, depth_frame2)

# Save point clouds to PLY files
save_point_cloud_to_ply(point_cloud1, ply_file1)
save_point_cloud_to_ply(point_cloud2, ply_file2)

print(f"Point clouds saved to {ply_file1} and {ply_file2}")
