import open3d as o3d

def load_point_cloud(ply_filename):
    return o3d.io.read_point_cloud(ply_filename)

def remove_isolated_points(point_cloud, nb_neighbors, std_ratio):
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = point_cloud.select_by_index(ind)
    return inlier_cloud

def save_point_cloud(ply_filename, point_cloud):
    o3d.io.write_point_cloud(ply_filename, point_cloud)

def visualize_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])

# Load point cloud from PLY file
input_ply_file = 'merged_cloud.ply'
cleaned_ply_file = 'cleaned_cloud_boxfinal.ply'
cloud = load_point_cloud(input_ply_file)

if cloud.is_empty():
    print("Chmura punktów jest pusta.")
else:
    # Remove isolated points
    cleaned_cloud = remove_isolated_points(cloud, nb_neighbors=80, std_ratio=0.9)

    # Save cleaned point cloud to a new PLY file
    save_point_cloud(cleaned_ply_file, cleaned_cloud)
    print(f"Cleaned point cloud saved as '{cleaned_ply_file}'")

    # Visualize the cleaned point cloud
    visualize_point_cloud(cleaned_cloud)
