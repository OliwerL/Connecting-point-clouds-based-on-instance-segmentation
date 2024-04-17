import pyrealsense2 as rs
import numpy as np
import cv2

# Inicjalizacja pipeline
pipeline = rs.pipeline()
config = rs.config()

# Aktywacja strumienia kolorowego
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start przesyłania danych
pipeline.start(config)

try:
    while True:
        # Czekaj na klatkę
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Konwersja obrazu do numpy array
        img = np.asanyarray(color_frame.get_data())

        # Odczytanie intrynsekcji z profilu strumienia
        intr = color_frame.profile.as_video_stream_profile().intrinsics

        # Stworzenie macierzy kamery dla OpenCV
        camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                  [0, intr.fy, intr.ppy],
                                  [0, 0, 1]])
        dist_coeffs = np.array(intr.coeffs)  # Uwaga: sprawdź, czy wymiary zgadzają się z oczekiwaniami OpenCV
        print(f"fx = {intr.fx}")
        print(f"fy = {intr.fy}")
        print(f"cx = {intr.ppx}")
        print(f"xy = {intr.ppy}")
        print(f"initial_dist_coeffs = {dist_coeffs}")

        # Korygowanie zniekształceń
        img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

        # Wyświetlanie obrazu
        cv2.imshow('Undistorted Image', img_undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Zakończenie strumienia
    pipeline.stop()
    cv2.destroyAllWindows()