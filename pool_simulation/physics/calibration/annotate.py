import cv2
import numpy as np
import json
from pool_simulation.constants import TABLE_WIDTH, TABLE_HEIGHT, CUSHION_WIDTH

VISUAL_W, VISUAL_H = 1000, 500
current_click = None


def get_transform_matrix(corners_px, table_width_m, table_height_m):
    outer_x = (table_width_m / 2.0) + CUSHION_WIDTH
    outer_y = (table_height_m / 2.0) + CUSHION_WIDTH
    dst_pts = np.array([
        [-outer_x, outer_y], [0.0, outer_y],
        [outer_x, -outer_y], [-outer_x, -outer_y]
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners_px, dst_pts)


def mouse_callback(event, x, y, flags, param):
    global current_click
    if event == cv2.EVENT_LBUTTONDOWN:
        current_click = (x, y)


def annotate_video(video_path, corners_px, output_json, frame_skip=10, fps=60):
    global current_click

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}. Check the file path!")
        return

    physics_matrix = get_transform_matrix(corners_px, TABLE_WIDTH, TABLE_HEIGHT)

    visual_dst_pts = np.array([[0, 0], [VISUAL_W/2, 0], [VISUAL_W, VISUAL_H], [0, VISUAL_H]], dtype=np.float32)
    visual_matrix = cv2.getPerspectiveTransform(corners_px, visual_dst_pts)

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", mouse_callback)

    tracked_data = []
    frame_idx = 0

    # Read the very first frame to kick things off
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Video {video_path} is empty or unreadable.")
        return

    print(f"\n--- Annotating {video_path} ---")
    print("CONTROLS:")
    print(" [ n ] - Step forward exactly 1 frame (Find the exact impact!)")
    print(" [ s ] - Skip forward 10 frames (Fast forward through aiming)")
    print(" [ CLICK ] - Record ball position, automatically skips forward 10 frames")
    print(" [ q ] - Quit and save data\n")

    while True:
        warped_frame = cv2.warpPerspective(frame, visual_matrix, (VISUAL_W, VISUAL_H))
        current_click = None, None

        # Wait for user input on the CURRENT frame
        while True:
            display = warped_frame.copy()
            for data in tracked_data[-5:]:
                cv2.circle(display, data['pixel'], 4, (0, 0, 255), -1)

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
            elif current_click is not None:
                action = 'clicked'
                break

        if action == 'quit':
            break

        elif action == 'clicked':
            px, py = current_click
            pt_warped = np.array([[[px, py]]], dtype=np.float32)
            inv_visual = np.linalg.inv(visual_matrix)
            pt_raw = cv2.perspectiveTransform(pt_warped, inv_visual)
            pt_m = cv2.perspectiveTransform(pt_raw, physics_matrix)

            physics_x, physics_y = pt_m[0][0]
            timestamp = frame_idx / fps

            tracked_data.append({
                'time': timestamp, 'x': float(physics_x), 'y': float(physics_y), 'pixel': (px, py)
            })

            print(f"Recorded T={timestamp:.3f}s | Pos: ({physics_x:.3f}, {physics_y:.3f})")
            action = 'skip_10'  # Automatically skip forward after a click

        # Silently advance the video stream by the correct number of frames
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
    clean_data = [{'time': d['time'], 'x': d['x'], 'y': d['y']} for d in data]
    with open(filename, 'w') as f:
        json.dump(clean_data, f, indent=4)
    print(f"\nSaved {len(clean_data)} points to {filename}!")


if __name__ == "__main__":
    corners_1 = np.array([[189, 149], [1236, 135], [2482, 1221], [13, 1306]], dtype=np.float32)
    corners_3 = np.array([[134, 176], [1173, 118], [2430, 1138], [19, 1323]], dtype=np.float32)

    annotate_video("20260226_201007_1.mp4", corners_1, "shot1_data.json", frame_skip=10)
    annotate_video("20260226_201007_3.mp4", corners_3, "shot3_data.json", frame_skip=10)