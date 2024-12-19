import os
import zipfile
import shutil
import yaml
import random

# 현재 코드 파일의 디렉토리 경로
base_dir = os.path.dirname(os.path.abspath(__file__))

# Dataset URLs
dataset_urls = [
    "https://universe.roboflow.com/ds/BqZyOqmham?key=bF5x6RQQxY",
    "https://universe.roboflow.com/ds/Wlnrekyqfx?key=4TFDuvEo6S",
    "https://universe.roboflow.com/ds/LHGWWQsIRx?key=31DCfAN77S",
    "https://universe.roboflow.com/ds/MhGNV9TTAW?key=YwLDmOfWuO",
    "https://universe.roboflow.com/ds/5Bw5XAOSZC?key=l9bKhsQ2gp"
]

# Class mapping
class_mapping = {
    "original_class_1": "hammer",
    "original_class_2": "rubber_mallet",
    "original_class_3": "Palu",
    "original_class_4": "Palu",
    "original_class_5": "hammer"
}

# Paths
dataset_dir = os.path.join(base_dir, "datasets")
unified_dataset_dir = os.path.join(base_dir, "hammer_dataset")
train_images_path = os.path.join(unified_dataset_dir, "train/images")
train_labels_path = os.path.join(unified_dataset_dir, "train/labels")
val_images_path = os.path.join(unified_dataset_dir, "valid/images")
val_labels_path = os.path.join(unified_dataset_dir, "valid/labels")

os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# Download and extract datasets
def download_and_extract_datasets():
    os.makedirs(dataset_dir, exist_ok=True)
    for idx, url in enumerate(dataset_urls):
        zip_path = os.path.join(dataset_dir, f"dataset_{idx + 1}.zip")
        extract_path = os.path.join(dataset_dir, f"dataset_{idx + 1}")
        print(f"Downloading dataset {idx + 1}...")
        os.system(f"wget -O {zip_path} {url}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

# Unify and split datasets
def unify_datasets():
    dataset_paths = [os.path.join(dataset_dir, f"dataset_{i + 1}") for i in range(len(dataset_urls))]
    all_images = []
    all_labels = []

    for dataset_path in dataset_paths:
        images_path = os.path.join(dataset_path, "train/images")  # 예상 경로
        labels_path = os.path.join(dataset_path, "train/labels")  # 예상 경로

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Warning: {images_path} or {labels_path} does not exist. Skipping...")
            continue

        images = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))])
        labels = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt')])

        valid_files = set([f.split('.')[0] for f in images]) & set([f.split('.')[0] for f in labels])
        images = [f for f in images if f.split('.')[0] in valid_files]
        labels = [f for f in labels if f.split('.')[0] in valid_files]

        for img_file, lbl_file in zip(images, labels):
            all_images.append(os.path.join(images_path, img_file))
            all_labels.append(os.path.join(labels_path, lbl_file))

    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    split_index = int(len(all_images) * 0.8)
    train_images, val_images = all_images[:split_index], all_images[split_index:]
    train_labels, val_labels = all_labels[:split_index], all_labels[split_index:]

    for img, lbl in zip(train_images, train_labels):
        shutil.copy(img, train_images_path)
        shutil.copy(lbl, train_labels_path)

    for img, lbl in zip(val_images, val_labels):
        shutil.copy(img, val_images_path)
        shutil.copy(lbl, val_labels_path)

# Execute the steps
download_and_extract_datasets()
unify_datasets()

# YAML configuration for YOLO
data = {
    'train': train_images_path,
    'val': val_images_path,
    'names': ['hammer'],
    'nc': 1
}

yaml_path = os.path.join(unified_dataset_dir, "hammer_data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(data, f)

print("Generated YAML file:")
with open(yaml_path, 'r') as f:
    print(f.read())
