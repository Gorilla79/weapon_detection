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

# Depth correction factor (adjusted to your calibration)
SCALING_FACTOR = 1.119  # To correct depth measurement
OFFSET = 0.0  # Additional offset correction in meters (if needed)

# Load YOLO model
device = 'cuda'
model = YOLO(r"D:\\capstone\\24_12_03\\best.pt").to(device)
names = model.names

# Output folders
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

# Tracking and targeting variables
track_history = defaultdict(lambda: [])
frame_counter = 0
interested_object_id = None

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    im0 = np.asanyarray(color_frame.get_data())

    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
    clss = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

    annotator = Annotator(im0)

    # Annotate all detections
    for box, cls, track_id in zip(boxes, clss, track_ids):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Get raw depth and apply correction
        raw_depth = depth_frame.get_distance(center_x, center_y)
        corrected_depth = raw_depth * SCALING_FACTOR + OFFSET

        depth_str = f"Depth: {corrected_depth:.2f} meters"
        label = f"ID:{track_id} {names[int(cls)]} {results[0].boxes.conf[0]:.2f}"

        print(f"ID {track_id} - Center: ({center_x}, {center_y}), Corrected Depth: {corrected_depth:.2f} meters")

        # Annotate bounding boxes
        annotator.box_label((x1, y1, x2, y2), label, color=(0, 0, 255))
        cv2.putText(im0, depth_str, (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display annotated frame
    cv2.imshow("YOLOv8 Detection with Depth Correction", annotator.result())

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()