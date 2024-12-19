import os
from ultralytics import YOLO

def train_yolo():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "hammer_dataset", "hammer_data.yaml")

    # Verify YAML file exists
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} does not exist. Make sure the dataset preparation step completed successfully.")
        return

    # Train YOLO model
    try:
        print("Starting YOLOv8 model training...")
        model = YOLO('yolov8n.pt')  # Load the pretrained YOLOv8 nano model
        model.train(
            data=yaml_path,
            epochs=100,
            batch=32,
            imgsz=416,
            name="hammer_training",
            patience=30  # Early stopping if no improvement for 30 epochs
        )
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == '__main__':
    train_yolo()
