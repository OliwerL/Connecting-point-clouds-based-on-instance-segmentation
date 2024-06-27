import numpy as np
import open3d as o3d
import json
import cv2

def load_point_cloud(ply_filename):
    return o3d.io.read_point_cloud(ply_filename)

def apply_transformation(point_cloud, R, T):
    R = np.array(R)
    T = np.array(T)/1000
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T.reshape(-1)
    point_cloud.transform(transformation_matrix)
    return point_cloud

def merge_point_clouds(clouds):
    merged_cloud = clouds[0]
    for cloud in clouds[1:]:
        merged_cloud += cloud
    return merged_cloud
# Load calibration data
with open('glowna_karton_prawa_karton_STEREO.json', 'r') as f:
    calibration_data1 = json.load(f)

R1 = calibration_data1['R']
T1 = calibration_data1['T']

with open('glowna_karton_lewa_karton_STEREO.json', 'r') as f:
    calibration_data2 = json.load(f)

R2 = calibration_data2['R']
T2 = calibration_data2['T']

# Load point clouds
cloud1 = load_point_cloud('filtered_cloud2.ply')
cloud2 = load_point_cloud('filtered_cloud4.ply')
cloud3 = load_point_cloud('filtered_cloud.ply')

# Apply transformation to the second point cloud
cloud2_transformed = apply_transformation(cloud2, R1, T1)
cloud3_transformed = apply_transformation(cloud3, R2, T2)


# Merge point clouds
merged_cloud = merge_point_clouds([cloud1, cloud2_transformed, cloud3_transformed])

# Save merged point cloud to a new PLY file
o3d.io.write_point_cloud('merged_cloud.ply', merged_cloud)

print("Point clouds merged successfully. Output saved as 'merged_cloud.ply'")

# Ścieżka do pliku PLY
# ply_file = 'prawyPLY.ply'
ply_file = 'merged_cloud.ply'
def visualize_point_cloud(ply_file):
    # Wczytaj chmurę punktów z pliku PLY
    pcd = o3d.io.read_point_cloud(ply_file)

    # Sprawdź, czy chmura punktów została poprawnie wczytana
    if pcd.is_empty():
        print("Chmura punktów jest pusta.")
        return

    # Wyświetl chmurę punktów
    o3d.visualization.draw_geometries([pcd])

# Wizualizacja chmury punktów
visualize_point_cloud(ply_file)
