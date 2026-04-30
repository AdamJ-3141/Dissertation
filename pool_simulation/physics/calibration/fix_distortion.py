import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture("Shot_1.mp4")
ret, frame = cap.read()
if not ret:
    print("Could not read video.")
    exit()

h, w = frame.shape[:2]

cv2.namedWindow("Lens Correction", cv2.WINDOW_NORMAL)

cv2.createTrackbar("Focal Length", "Lens Correction", 50, 150, nothing)
cv2.createTrackbar("k1 (Primary Curve)", "Lens Correction", 100, 200, nothing)
cv2.createTrackbar("k2 (Corner Hook)", "Lens Correction", 100, 200, nothing)

# NEW: Principal Point Offsets (Shifting the optical center)
cv2.createTrackbar("Center X Offset", "Lens Correction", 100, 200, nothing)
cv2.createTrackbar("Center Y Offset", "Lens Correction", 100, 200, nothing)

cv2.createTrackbar("Grid Density", "Lens Correction", 15, 50, nothing)

print("--- ASYMMETRIC CALIBRATION ---")
print("1. Adjust 'Focal Length' to set the base scale.")
print("2. Adjust 'k1' until ONE cushion (e.g., the bottom) is perfectly straight.")
print(
    "3. If the top is now bowed opposite to the bottom, adjust 'Center Y Offset' to shift the distortion gravity up or down.")
print("4. Re-tweak k1 and Center Y together until both top and bottom are straight.")
print("Press 's' to save the corrected image.")
print("Press 'q' to quit.")

while True:
    f_mult = cv2.getTrackbarPos("Focal Length", "Lens Correction") / 100.0
    if f_mult == 0: f_mult = 0.01
    focal_length = w * f_mult

    # NEW: Calculate the shifted optical center
    # (100 is dead center. Sliding left/right shifts the center up to 20% of the image width/height)
    cx_shift = (cv2.getTrackbarPos("Center X Offset", "Lens Correction") - 100) / 100.0 * (w * 0.2)
    cy_shift = (cv2.getTrackbarPos("Center Y Offset", "Lens Correction") - 100) / 100.0 * (h * 0.2)

    cx = (w / 2) + cx_shift
    cy = (h / 2) + cy_shift

    cam_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    k1 = (cv2.getTrackbarPos("k1 (Primary Curve)", "Lens Correction") - 100) / 500.0
    k2 = (cv2.getTrackbarPos("k2 (Corner Hook)", "Lens Correction") - 100) / 500.0

    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

    undistorted = cv2.undistort(frame, cam_matrix, dist_coeffs)

    grid_density = max(1, cv2.getTrackbarPos("Grid Density", "Lens Correction"))
    step = h // grid_density

    for i in range(0, h, step):
        cv2.line(undistorted, (0, i), (w, i), (0, 255, 0), 1)
    for i in range(0, w, step):
        cv2.line(undistorted, (i, 0), (i, h), (0, 255, 0), 1)

    # Draw a red crosshair to show where the math thinks the optical center is
    cv2.drawMarker(undistorted, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.imshow("Lens Correction", undistorted)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('s'):
        cv2.imwrite("corrected_frame.jpg", undistorted)
        print(f"\nSaved 'corrected_frame.jpg'!")
        print(f"Cam Matrix: Focal={focal_length:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        print(f"Distortion Coeffs: k1={k1:.4f}, k2={k2:.4f}")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()