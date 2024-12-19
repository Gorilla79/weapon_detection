import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import time

# RealSense pipeline initialization
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load YOLO models
device = 'cuda'
model_human = YOLO("yolov8n.pt").to(device)  # Human detection model
model_weapon = YOLO(r"D:\\capstone\\24_12_18\\best_weapon.pt").to(device)  # Weapon detection model
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
frame_counter = 0
interested_object_id = None
proximity_threshold = 300  # Proximity threshold in pixels
danger_threshold = 1.5  # Time in seconds to consider as dangerous
overlap_start_times = {}

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    im0 = np.asanyarray(color_frame.get_data())

    # Run YOLO models
    results_human = model_human.track(im0, classes=[0], persist=False, device=device)  # Detect humans
    results_weapon = model_weapon.track(im0, classes=[0], persist=False, device=device)  # Detect weapons

    # Filter detections by confidence
    boxes_human = results_human[0].boxes.xyxy.cpu().numpy() if results_human[0].boxes else []
    ids_human = results_human[0].boxes.id.int().cpu().tolist() if results_human[0].boxes.id is not None else []

    if results_weapon[0].boxes:
        weapon_boxes = results_weapon[0].boxes
        weapon_confidences = weapon_boxes.conf.cpu().numpy()
        boxes_weapon = weapon_boxes.xyxy.cpu().numpy()[weapon_confidences >= 0.7]  # Confidence threshold: 70%
    else:
        boxes_weapon = []

    annotator = Annotator(im0)
    current_time = time.time()

    # Proximity logic: Check distance between humans and weapons
    for human_box, human_id in zip(boxes_human, ids_human):
        human_center = [(human_box[0] + human_box[2]) / 2, (human_box[1] + human_box[3]) / 2]

        for weapon_box in boxes_weapon:
            weapon_center = [(weapon_box[0] + weapon_box[2]) / 2, (weapon_box[1] + weapon_box[3]) / 2]
            distance = np.linalg.norm(np.array(human_center) - np.array(weapon_center))

            if distance < proximity_threshold:  # Close proximity
                if human_id not in overlap_start_times:
                    overlap_start_times[human_id] = current_time  # Start overlap timer

                elapsed_time = current_time - overlap_start_times[human_id]
                if elapsed_time >= danger_threshold:  # Target the human if danger persists
                    interested_object_id = human_id  # Set the current human as the target
                    x1, y1, x2, y2 = map(int, human_box)
                    center_x, center_y = int(human_center[0]), int(human_center[1])
                    depth = depth_frame.get_distance(center_x, center_y)
                    depth_str = f"Depth: {depth:.2f}m"

                    # Highlight the target
                    print(f"위험 감지: 사람 ID {human_id}, 중첩 지속 시간: {elapsed_time:.2f}초")
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(im0, f"TARGET: ID {human_id}", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(im0, depth_str, (x1, y1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                # Only reset the timer for IDs that are not the target
                if human_id != interested_object_id:
                    overlap_start_times.pop(human_id, None)

    # Annotate all detections
    for box, human_id in zip(boxes_human, ids_human):
        x1, y1, x2, y2 = map(int, box)
        if human_id == interested_object_id:  # Highlight the target
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for target
            cv2.putText(im0, f"TARGET: ID {human_id}", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            annotator.box_label((x1, y1, x2, y2), "Human", color=(0, 255, 0))  # Green box

    for box in boxes_weapon:
        x1, y1, x2, y2 = map(int, box)
        annotator.box_label((x1, y1, x2, y2), "Weapon", color=(0, 0, 255))  # Red box

    # Show annotated frame
    cv2.imshow("Dual Model Detection", annotator.result())

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
