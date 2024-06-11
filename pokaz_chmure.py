import open3d as o3d
import numpy as np

def visualize_point_cloud(ply_file):
    # Wczytaj chmurę punktów z pliku PLY
    pcd = o3d.io.read_point_cloud(ply_file)

    # Sprawdź, czy chmura punktów została poprawnie wczytana
    if pcd.is_empty():
        print("Chmura punktów jest pusta.")
        return

    # Wyświetl chmurę punktów
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Ścieżka do pliku PLY
    # ply_file = 'prawyPLY.ply'
    ply_file = 'glowna_karton_PLY.ply'
    # Wizualizacja chmury punktów
    visualize_point_cloud(ply_file)
