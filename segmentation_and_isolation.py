import open3d as o3d
import numpy as np

def segment_plane(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=4000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud, plane_model

def main():
    # Load Your Point Cloud
    pcd = o3d.io.read_point_cloud("merged_cloud.ply")
    print("Loaded point cloud:")

    # Preprocess the Point Cloud (Downsampling)
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.01)  # Smaller voxel size
    print("Downsampled point cloud:")
    o3d.visualization.draw_geometries([pcd_downsampled])

    # Step 1: Segment the largest plane (assumed to be the floor)
    max_planes = 3
    distance_threshold = 0.02
    floor_cloud = o3d.geometry.PointCloud()
    remaining_cloud = pcd_downsampled
    for _ in range(max_planes):
        plane, remaining_cloud, _ = segment_plane(remaining_cloud, distance_threshold=distance_threshold)
        floor_cloud += plane
        print(f"Segmented plane with {len(plane.points)} points")

    floor_cloud.paint_uniform_color([1.0, 0, 0])  # Red for the floor
    print("Segmented floor:")
    o3d.visualization.draw_geometries([floor_cloud])

    # Step 2: Cluster the remaining points to identify objects
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(remaining_cloud.cluster_dbscan(eps=0.039, min_points=70, print_progress=True))

    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")

    # Step 3: Color each cluster blue
    colors = np.tile([0, 0, 1.0], (len(labels), 1))  # Blue for all clusters
    colors[labels < 0] = 0
    remaining_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Step 4: Find and color the object at the center of the floor in green
    floor_center = np.mean(np.asarray(floor_cloud.points), axis=0)
    object_centers = []
    for i in range(max_label + 1):
        cluster_points = remaining_cloud.select_by_index(np.where(labels == i)[0]).points
        cluster_center = np.mean(np.asarray(cluster_points), axis=0)
        object_centers.append((i, cluster_center))

    # Find the cluster closest to the center of the floor
    distances = [np.linalg.norm(center - floor_center) for _, center in object_centers]
    closest_cluster_idx = object_centers[np.argmin(distances)][0]

    # Color the closest cluster green
    object_indices = np.where(labels == closest_cluster_idx)[0]
    remaining_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Reset colors
    np.asarray(remaining_cloud.colors)[object_indices] = [0, 1.0, 0]  # Green for the central object

    print("Clustered remaining points (objects):")
    o3d.visualization.draw_geometries([floor_cloud, remaining_cloud])

    # Extract and display the green object
    green_object_cloud = remaining_cloud.select_by_index(object_indices)
    print("Displaying the green object:")
    o3d.visualization.draw_geometries([green_object_cloud])

if __name__ == "__main__":
    main()
