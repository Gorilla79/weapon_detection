import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch

# RealSense pipeline initialization
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load YOLO models
device = 'cuda'
model_human = YOLO("yolov8n.pt").to(device)
model_weapon_hammer = YOLO(r"D:\\capstone\\25_01_07\\best_hammer.pt").to(device)
model_weapon_knife = YOLO(r"D:\\capstone\\25_01_07\\best_knife.pt").to(device)
model_weapon_gun = YOLO(r"D:\\capstone\\25_01_07\\best_gun.pt").to(device)
model_weapon_bat = YOLO(r"D:\\capstone\\25_01_07\\best_bat.pt").to(device)
model_weapon_axe = YOLO(r"D:\\capstone\\25_01_07\\best_axe.pt").to(device)
print("Models loaded on CUDA")

# Weapon classes for labeling
weapon_classes = {
    "hammer": model_weapon_hammer.names[0],  # Hammer class name
    "knife": model_weapon_knife.names[0],   # Knife class name
    "gun": model_weapon_gun.names[0],       # Gun class name
    "bat": model_weapon_bat.names[0],       # Bat class name
    "axe": model_weapon_axe.names[0],       # Bat class name
}

# Constants
proximity_threshold = 300  # Pixels
confidence_threshold = 0.6  # Minimum confidence threshold
overlap_threshold = 0.05  # Minimum 10% overlap for dangerous determination
frame_threshold = 20  # Frames for persistent dangerous status

# Intrinsic parameters for alignment
align = rs.align(rs.stream.color)

# Dangerous tracking variables
frame_counters = {}
target_dangerous_id = None  # ID of the currently targeted dangerous person

# Mouse callback for depth measurement
clicked_point = None
clicked_depth = None


def get_depth_at_click(event, x, y, flags, param):
    global clicked_point, clicked_depth
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.namedWindow("Color and Depth Detection")
cv2.setMouseCallback("Color and Depth Detection", get_depth_at_click)

# Function to calculate overlap ratio
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

# Adjust text position to stay within image bounds
def adjust_text_position_side(x, y, texts, box, image_shape):
    positions = []
    text_height = 15  # Approximate height of one line of text

    for i, text in enumerate(texts):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_width, _ = text_size

        # Default position
        adjusted_x = x
        adjusted_y = y + i * text_height

        # Check if text exceeds image boundaries
        if adjusted_x + text_width > image_shape[1]:
            # Shift text to the left of the bounding box
            adjusted_x = box[0] - text_width - 10
        if adjusted_y - text_height < 0:
            adjusted_y = text_height + 10 + i * text_height

        positions.append((adjusted_x, adjusted_y))
    return positions

def process_weapon_detections(results_hammer, results_knife, results_gun, results_bat, results_axe, confidence_threshold):
    all_boxes = []
    all_classes = []
    all_confidences = []

    # Collect detections from all models
    models = [
        (results_hammer, "Hammer", 0.8),
        (results_knife, "Knife", 0.6),
        (results_gun, "Gun", 0.7),
        (results_bat, "Bat", 0.5),
        (results_axe, "Axe", 0.6),
    ]

    for result, class_name, threshold in models:
        if result[0].boxes:
            boxes = result[0].boxes.xyxy.cpu().numpy()
            confidences = result[0].boxes.conf.cpu().numpy()
            filtered_indices = np.where(confidences >= threshold)[0]
            boxes = boxes[filtered_indices]
            confidences = confidences[filtered_indices]

            all_boxes.extend(boxes)
            all_classes.extend([class_name] * len(filtered_indices))
            all_confidences.extend(confidences)

    if len(all_boxes) == 0:
        return np.empty((0, 4)), [], []

    # Convert to numpy array
    all_boxes = np.array(all_boxes)
    all_confidences = np.array(all_confidences)
    all_classes = np.array(all_classes)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=all_boxes.tolist(),
        scores=all_confidences.tolist(),
        score_threshold=confidence_threshold,
        nms_threshold=0.5,  # Adjust overlap threshold as needed
    )

    if len(indices) > 0:
        indices = indices.flatten()  # Flatten indices to a list
        all_boxes = all_boxes[indices]
        all_classes = all_classes[indices]
        all_confidences = all_confidences[indices]
    else:
        all_boxes = np.empty((0, 4))
        all_classes = []
        all_confidences = []

    return all_boxes, all_classes, all_confidences

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 이미지 중심 좌표 계산
        height, width = color_image.shape[:2]
        cx, cy = width // 2, height // 2

        # Normalize depth image for display
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # Run YOLO detection
        results_human = model_human.track(color_image, classes=[0], persist=True, device=device)
        results_hammer = model_weapon_hammer.track(color_image, persist=True, device=device)
        results_knife = model_weapon_knife.track(color_image, persist=True, device=device)
        results_gun = model_weapon_gun.track(color_image, persist=True, device=device)
        results_bat = model_weapon_bat.track(color_image, persist=True, device=device)
        results_axe = model_weapon_axe.track(color_image, persist=True, device=device)

        # Process weapon detection results
        boxes_weapon, classes_weapon, confidences_weapon = process_weapon_detections(
            results_hammer, results_knife, results_gun, results_bat, results_axe, confidence_threshold
        )

        print("Detected weapons:", classes_weapon)

        # Extract human detections
        boxes_human = []
        ids_human = []
        if results_human[0].boxes:
            human_boxes = results_human[0].boxes
            human_confidences = human_boxes.conf.cpu().numpy()
            boxes_human = human_boxes.xyxy.cpu().numpy()[human_confidences >= confidence_threshold]
            ids_human = (
                human_boxes.id.cpu().numpy()[human_confidences >= confidence_threshold]
                if human_boxes.id is not None
                else []
            )

        '''
        # Extract weapon detections
        boxes_weapon = []
        classes_weapon = []
        confidences_weapon = []

        if results_weapon[0].boxes:
            weapon_boxes = results_weapon[0].boxes
            weapon_confidences = weapon_boxes.conf.cpu().numpy()
            weapon_classes = weapon_boxes.cls.cpu().numpy()

            # Filter by confidence threshold
            filtered_indices = np.where(weapon_confidences >= confidence_threshold)[0]
            boxes_weapon = weapon_boxes.xyxy.cpu().numpy()[filtered_indices]
            classes_weapon = weapon_classes[filtered_indices]
            confidences_weapon = weapon_confidences[filtered_indices]
        '''

        current_frame_human_ids = []

        # Annotate humans and check for dangerous status
        for box, human_id in zip(boxes_human, ids_human):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            depth = depth_frame.get_distance(center_x, center_y)

            # 중심 기준 좌표 계산
            centered_x = center_x - cx
            centered_y = center_y - cy

            is_dangerous = False

            # Check overlap with weapons (hammer or knife or gun or bat or axe)
            for weapon_box, weapon_class in zip(boxes_weapon, classes_weapon):
                overlap_ratio = calculate_overlap(box, weapon_box)
                if overlap_ratio >= overlap_threshold:
                    if human_id not in current_frame_human_ids:
                        current_frame_human_ids.append(human_id)  # 중복 없이 추가
                    if human_id not in frame_counters:
                        frame_counters[human_id] = 0
                    frame_counters[human_id] += 1  # 카운트 증가

                    if frame_counters[human_id] >= frame_threshold:
                        target_dangerous_id = human_id  # 위험 대상으로 갱신
                        is_dangerous = True
                    break

            # Label and color settings based on danger status
            if is_dangerous or human_id == target_dangerous_id:
                label = f"ID:{human_id}, Dangerous"
                color = (0, 0, 255)  # Red for dangerous
            else:
                label = f"ID:{human_id}, Human"
                color = (0, 255, 0)  # Green for non-dangerous
            depth_text = f"X:{centered_x}, Y:{centered_y}, Depth:{depth:.2f}m"

            # Draw bounding box
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)

            # Adjust text position and output
            positions = adjust_text_position_side(
                x1, y1 - 2, [label, depth_text], (x1, y1, x2, y2), color_image.shape
            )
            for text, (pos_x, pos_y) in zip([label, depth_text], positions):
                cv2.putText(color_image, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Mark center point
            cv2.circle(color_image, (center_x, center_y), 10, color, -1)

            # Add to depth map
            adjusted_x = min(max(center_x, 10), depth_colormap.shape[1] - 50)
            adjusted_y = min(max(center_y, 20), depth_colormap.shape[0] - 10)

            # 중심 기준 좌표로 변경
            centered_x = adjusted_x - cx
            centered_y = adjusted_y - cy

            cv2.circle(depth_colormap, (adjusted_x, adjusted_y), 10, color, -1)

            depth_label = f"X:{center_x}, Y:{center_y}, Depth:{depth:.2f}m"
            depth_label_positions = adjust_text_position_side(
                adjusted_x + 10, adjusted_y - 10, [depth_label], (adjusted_x, adjusted_y, adjusted_x, adjusted_y), depth_colormap.shape
            )
            for text, (pos_x, pos_y) in zip([depth_label], depth_label_positions):
                cv2.putText(depth_colormap, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Annotate weapon detections
        for box, weapon_class, confidence in zip(boxes_weapon, classes_weapon, confidences_weapon):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            depth = depth_frame.get_distance(center_x, center_y)

            # 중심 기준 좌표 계산
            centered_x = center_x - cx
            centered_y = center_y - cy

            # Draw bounding box
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Annotate class name and confidence
            label = f"{weapon_class} ({confidence:.2f}), Depth:{depth:.2f}m"
            cv2.putText(
                color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )

            # Add to depth map
            adjusted_x = min(max(center_x, 10), depth_colormap.shape[1] - 50)
            adjusted_y = min(max(center_y, 20), depth_colormap.shape[0] - 10)
            cv2.circle(depth_colormap, (adjusted_x, adjusted_y), 10, (0, 255, 255), -1)

            weapon_label = f"{weapon_class} ({confidence:.2f}), Depth:{depth:.2f}m"
            weapon_label_positions = adjust_text_position_side(
                adjusted_x + 10, adjusted_y - 10, [weapon_label], (adjusted_x, adjusted_y, adjusted_x, adjusted_y), depth_colormap.shape
            )
            for text, (pos_x, pos_y) in zip([weapon_label], weapon_label_positions):
                cv2.putText(depth_colormap, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Annotate clicked point
        if clicked_point:
            click_x, click_y = clicked_point
            clicked_depth = depth_frame.get_distance(click_x, click_y)
            
            #중심점 좌표 계산
            centered_click_x = click_x - cx
            centered_click_y = click_y - cy

            adjusted_x = min(max(click_x, 10), depth_colormap.shape[1] - 50)
            adjusted_y = min(max(click_y, 20), depth_colormap.shape[0] - 10)
            cv2.circle(color_image, (click_x, click_y), 10, (0, 255, 255), -1)
            cv2.circle(depth_colormap, (adjusted_x, adjusted_y), 10, (0, 255, 255), -1)

            click_label = f"X:{centered_click_x}, Y:{centered_click_y}, Depth:{clicked_depth:.2f}m"
            click_label_positions = adjust_text_position_side(
                adjusted_x + 10, adjusted_y - 10, [click_label], (adjusted_x, adjusted_y, adjusted_x, adjusted_y), depth_colormap.shape
            )
            for text, (pos_x, pos_y) in zip([click_label], click_label_positions):
                cv2.putText(depth_colormap, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Stack images side-by-side
        combined_image = np.hstack((color_image, depth_colormap))
        cv2.imshow("Color and Depth Detection", combined_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            clicked_point = None
            frame_counters.clear()
            target_dangerous_id = None

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
