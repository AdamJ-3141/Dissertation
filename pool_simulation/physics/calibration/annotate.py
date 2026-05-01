import cv2
import numpy as np
import json
from pool_simulation.constants import TABLE_WIDTH, TABLE_HEIGHT, CUSHION_WIDTH

VISUAL_W, VISUAL_H = 1000, 500
current_click = None
active_ball_id = 0  # Default to Cue Ball (0)


def get_transform_matrix(corners_px, table_width_m, table_height_m):
    # Half-dimensions of the playing surface
    w = table_width_m / 2.0
    h = table_height_m / 2.0

    dst_pts = np.array([
        [-w, h],  # Top-Left
        [w, h],  # Top-Right
        [0.0, -h],  # Bottom-Middle
        [-w, -h]  # Bottom-Left
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(corners_px, dst_pts)


def mouse_callback(event, x, y, flags, param):
    global current_click
    if event == cv2.EVENT_LBUTTONDOWN:
        current_click = (x, y)


def annotate_video(video_path, corners_px, output_json, frame_skip=10, fps=60):
    global current_click, active_ball_id

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}. Check the file path!")
        return

    # Read the first frame to get the video dimensions
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Video {video_path} is empty or unreadable.")
        return

    h, w = frame.shape[:2]

    # The raw slider values
    f_slider = 63
    k1_slider = 113
    k2_slider = 77
    cx_slider = 64
    cy_slider = 0

    # Translating sliders into OpenCV Camera Matrix math
    focal_length = w * (f_slider / 100.0)
    cx_shift = ((cx_slider - 100) / 100.0) * (w * 0.2)
    cy_shift = ((cy_slider - 100) / 100.0) * (h * 0.2)

    cx = (w / 2) + cx_shift
    cy = (h / 2) + cy_shift

    cam_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Translating sliders into OpenCV Distortion Coefficients
    k1 = (k1_slider - 100) / 500.0
    k2 = (k2_slider - 100) / 500.0
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

    physics_matrix = get_transform_matrix(corners_px, TABLE_WIDTH, TABLE_HEIGHT)
    visual_dst_pts = np.array([
        [0, 0],  # Top-Left corner of screen
        [VISUAL_W, 0],  # Top-Right corner of screen
        [VISUAL_W / 2.0, VISUAL_H],  # Bottom-Middle of screen
        [0, VISUAL_H]  # Bottom-Left corner of screen
    ], dtype=np.float32)
    visual_matrix = cv2.getPerspectiveTransform(corners_px, visual_dst_pts)

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", mouse_callback)

    tracked_data = []
    frame_idx = 0

    print(f"\n--- Annotating {video_path} ---")
    print("CONTROLS:")
    print(" [ 0-9 ] - Select Active Ball ID (0=Cue, 1=Target1, etc.)")
    print(" [ CLICK ] - Record position for the Active Ball")
    print(" [ n ] - Step forward exactly 1 frame")
    print(" [ s ] - Skip forward 10 frames")
    print(" [ q ] - Quit and save data\n")

    # Define some distinct colors for different ball IDs in the UI
    colors = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]

    while True:
        frame = cv2.undistort(frame, cam_matrix, dist_coeffs)
        warped_frame = cv2.warpPerspective(frame, visual_matrix, (VISUAL_W, VISUAL_H))
        current_click = None

        while True:
            display = warped_frame.copy()

            # Draw text indicating which ball is currently selected
            ui_color = colors[active_ball_id % len(colors)]
            cv2.putText(display, f"Active Ball ID: {active_ball_id}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, ui_color, 2)

            # Draw the last 15 points recorded to show trails
            for data in tracked_data[-15:]:
                b_color = colors[data['ball_id'] % len(colors)]
                cv2.circle(display, data['pixel'], 4, b_color, -1)

            cv2.imshow("Annotator", display)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                action = 'quit'
                break
            elif key == ord('n'):
                action = 'next_1'
                break
            elif key == ord('s'):
                action = 'skip_10'
                break
            elif ord('0') <= key <= ord('9'):
                active_ball_id = int(chr(key))
                # Do not break; let the user click the newly selected ball
            elif current_click is not None:
                # Process click immediately without advancing the frame
                px, py = current_click
                pt_warped = np.array([[[px, py]]], dtype=np.float32)
                inv_visual = np.linalg.inv(visual_matrix)
                pt_raw = cv2.perspectiveTransform(pt_warped, inv_visual)
                pt_m = cv2.perspectiveTransform(pt_raw, physics_matrix)

                physics_x, physics_y = pt_m[0][0]
                timestamp = frame_idx / fps

                tracked_data.append({
                    'time': timestamp,
                    'ball_id': active_ball_id,
                    'x': float(physics_x),
                    'y': float(physics_y),
                    'pixel': (px, py)
                })

                print(f"Recorded Ball {active_ball_id} | T={timestamp:.3f}s | Pos: ({physics_x:.3f}, {physics_y:.3f})")
                current_click = None  # Reset click so user can select another ball

        if action == 'quit':
            break

        frames_to_advance = 1 if action == 'next_1' else frame_skip

        for _ in range(frames_to_advance):
            ret, frame = cap.read()
            frame_idx += 1
            if not ret:
                break

        if not ret:
            print("End of video reached.")
            break

    cap.release()
    cv2.destroyAllWindows()
    save_data(output_json, tracked_data)


def save_data(filename, data):
    clean_data = [{'time': d['time'], 'ball_id': d['ball_id'], 'x': d['x'], 'y': d['y']} for d in data]
    with open(filename, 'w') as f:
        json.dump(clean_data, f, indent=4)
    print(f"\nSaved {len(clean_data)} points to {filename}!")


if __name__ == "__main__":
    # Locked tripod coordinates
    # Order: Top-Left, Top-Middle, Top-Right, Bottom-Middle
    tripod_reference_points = np.array([
        [230, 98],  # Top-Left
        [1753, 111],  # Top-Right
        [987, 936],  # Bottom-Middle
        [-9, 914]  # Bottom-Left
    ], dtype=np.float32)

    # Loop through all 8 videos automatically
    for i in [3]:
        video_filename = f"Shot_{i}.mp4"
        output_filename = f"shot{i}_data.json"

        # We wrap it in a try/except just in case one of the videos is missing
        try:
            annotate_video(video_filename, tripod_reference_points, output_filename, frame_skip=10)
        except Exception as e:
            print(f"Skipping {video_filename} due to error: {e}")

    print("\nAll videos processed!")