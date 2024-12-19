import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import time

# RealSense pipeline initialization
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load YOLO models
device = 'cuda'
model_human = YOLO("yolov8n.pt").to(device)
model_weapon = YOLO(r"D:\\capstone\\24_12_18\\best_weapon.pt").to(device)
print("Models loaded on CUDA")

# Create output directories
crop_dir_name = "capture-picture"
crop_dir_video = "capture-video"
def reset_folder(folder):
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    else:
        os.mkdir(folder)
reset_folder(crop_dir_name)
reset_folder(crop_dir_video)

# Initialize tracking variables
id_positions = {}
interested_object_id = None
proximity_threshold = 300  # Proximity threshold in pixels
frame_counters = {}  # Frame counters for overlap
frame_threshold = 15  # Frame threshold for Dangerous Human determination (~0.5 seconds at 30fps)

# Initialize video writer
video_writer = None
video_output_path = os.path.join(crop_dir_video, "dangerous_human.mp4")
def init_video_writer(frame_shape):
    global video_writer
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output_path, fourcc, 30, (frame_shape[1], frame_shape[0]))

# Overlap calculation function
def calculate_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    overlap_x_min = max(x1_min, x2_min)
    overlap_y_min = max(y1_min, y2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_max, y2_max)

    if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
        overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        return overlap_area / box1_area
    return 0

# Mapping 3D coordinates to 2D using Pythagoras
def map_3d_to_2d_pythagoras(center_x, center_y, depth_frame, screen_center):
    """
    Maps 3D depth information to 2D coordinates using Pythagoras.
    """
    # Screen center alignment
    x_offset = center_x - screen_center[0]
    y_center = center_y  # y remains as it is

    # Convert pixel distance (x_offset) to real-world distance using depth scaling
    depth_values = []
    for dy in range(-5, 6):  # y ± 5
        sampled_depth = depth_frame.get_distance(center_x, y_center + dy)
        if sampled_depth > 0:  # Ignore invalid depth
            depth_values.append(sampled_depth)
    
    if not depth_values:
        return None, None  # Invalid depth
    
    # Average depth for y ± 5 pixels
    avg_depth = sum(depth_values) / len(depth_values)
    
    # Convert x pixel offset to real-world x using depth scaling
    scale_factor = avg_depth / screen_center[0]  # Scale based on depth and screen width
    real_world_x = x_offset * scale_factor

    # Calculate the real-world y distance using Pythagoras
    real_world_y = (avg_depth**2 - real_world_x**2)**0.5 if avg_depth > abs(real_world_x) else None

    return real_world_x, real_world_y

# Update screen center for your system
screen_center = (320, 240)  # Assuming a 640x480 resolution

# Resize output window
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 1280, 720)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    im0 = np.asanyarray(color_frame.get_data())

    # Run YOLO models
    results_human = model_human.track(im0, classes=[0], persist=True, device=device)
    results_weapon = model_weapon.track(im0, classes=[0], persist=True, device=device)

    boxes_human = results_human[0].boxes.xyxy.cpu().numpy() if results_human[0].boxes else []
    ids_human = results_human[0].boxes.id.cpu().numpy() if results_human[0].boxes.id is not None else []

    if results_weapon[0].boxes:
        weapon_boxes = results_weapon[0].boxes
        weapon_confidences = weapon_boxes.conf.cpu().numpy()
        boxes_weapon = weapon_boxes.xyxy.cpu().numpy()[weapon_confidences >= 0.5]  # Confidence threshold: 70%
    else:
        boxes_weapon = []

    annotator = Annotator(im0)
    init_video_writer(im0.shape)

    # Reset frame counters for humans not in overlap
    current_frame_human_ids = []

    # Check overlaps and set target
    for human_box, human_id in zip(boxes_human, ids_human):
        # Compute bounding box center
        center_x = int((human_box[0] + human_box[2]) / 2)
        center_y = int((human_box[1] + human_box[3]) / 2)
        depth = depth_frame.get_distance(center_x, center_y)

        # Map 3D depth to 2D
        real_world_x, real_world_y = map_3d_to_2d_pythagoras(center_x, center_y, depth_frame, screen_center)
        
        if real_world_x is not None and real_world_y is not None:
            label_coords = f"2D:X:{real_world_x:.2f} Y:{real_world_y:.2f}"
        else:
            label_coords = "2D: Invalid Depth"

        for weapon_box in boxes_weapon:
            overlap_ratio = calculate_overlap(human_box, weapon_box)
            distance = np.linalg.norm(np.array([center_x, center_y]) - np.array([(weapon_box[0] + weapon_box[2]) / 2, (weapon_box[1] + weapon_box[3]) / 2]))

            if overlap_ratio > 0.2 or distance < proximity_threshold:  # Overlap or proximity threshold
                current_frame_human_ids.append(human_id)
                if human_id not in frame_counters:
                    frame_counters[human_id] = 0  # Initialize counter for this human
                frame_counters[human_id] += 1

                if frame_counters[human_id] > frame_threshold and interested_object_id is None:
                    interested_object_id = human_id

                    # Save images of the detected dangerous human and weapon
                    x1, y1, x2, y2 = map(int, human_box)
                    human_crop = im0[y1:y2, x1:x2]
                    human_path = os.path.join(crop_dir_name, f"dangerous_human_{human_id}.png")
                    cv2.imwrite(human_path, human_crop)

                    x1, y1, x2, y2 = map(int, weapon_box)
                    weapon_crop = im0[y1:y2, x1:x2]
                    weapon_path = os.path.join(crop_dir_name, f"weapon_{human_id}.png")
                    cv2.imwrite(weapon_path, weapon_crop)

    # Reset frame counters for humans not overlapping in this frame
    for human_id in list(frame_counters.keys()):
        if human_id not in current_frame_human_ids:
            del frame_counters[human_id]

    # Highlight only the targeted person
    if interested_object_id is not None:
        for human_box, human_id in zip(boxes_human, ids_human):
            if human_id == interested_object_id:
                center_x = int((human_box[0] + human_box[2]) / 2)
                center_y = int((human_box[1] + human_box[3]) / 2)
                depth = depth_frame.get_distance(center_x, center_y)

                real_world_x, real_world_y = map_3d_to_2d_pythagoras(center_x, center_y, depth_frame, screen_center)

                x1, y1, x2, y2 = map(int, human_box)
                label_id = f"DANGEROUS HUMAN ID:{human_id}"
                label_coords = f"2D:X:{real_world_x:.2f} Y:{real_world_y:.2f}" if real_world_x is not None and real_world_y is not None else "2D: Invalid Depth"
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box for target
                cv2.putText(im0, label_id, (x1, max(y1 - 30, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(im0, label_coords, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"{label_id} {label_coords}")
    else:
        for human_box, human_id in zip(boxes_human, ids_human):
            center_x = int((human_box[0] + human_box[2]) / 2)
            center_y = int((human_box[1] + human_box[3]) / 2)
            depth = depth_frame.get_distance(center_x, center_y)

            real_world_x, real_world_y = map_3d_to_2d_pythagoras(center_x, center_y, depth_frame, screen_center)

            x1, y1, x2, y2 = map(int, human_box)
            label_id = f"ID:{human_id}"
            label_coords = f"2D:X:{real_world_x:.2f} Y:{real_world_y:.2f}" if real_world_x is not None and real_world_y is not None else "2D: Invalid Depth"
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(im0, label_id, (x1, max(y1 - 30, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(im0, label_coords, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for weapon_box in boxes_weapon:
        x1, y1, x2, y2 = map(int, weapon_box)
        annotator.box_label((x1, y1, x2, y2), "Weapon", color=(0, 0, 255))

    # Write frame to video
    if video_writer:
        video_writer.write(im0)

    # Show frame
    cv2.imshow("Detection", annotator.result())

    # Exit or Reset
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset tracking variables
        interested_object_id = None
        frame_counters.clear()
        print("System reset.")

pipeline.stop()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
