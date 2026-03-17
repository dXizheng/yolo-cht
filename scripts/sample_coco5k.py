"""
COCO5K Dataset Preparation Script

This script:
1. Loads COCO annotations from instances_minitrain2017.json
2. Samples 5K images randomly
3. Downloads images from COCO server
4. Converts COCO bbox annotations to YOLO format
5. Creates the folder structure for YOLO training
"""

import json
import os
import random
import shutil
from pathlib import Path
import urllib.request
import ssl
import time

# Try to import tqdm, if not available use simple progress
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
ANNOTATIONS_FILE = PROJECT_DIR / "instances_minitrain2017.json"
DATASET_DIR = PROJECT_DIR / "coco5000"
SAMPLE_SIZE = 5000
VAL_SIZE = 500  # 500 for validation, 4500 for training


def load_annotations(annotations_file):
    """Load COCO annotations from JSON file."""
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data['images'])} images and {len(data['annotations'])} annotations")
    return data


def sample_images(data, sample_size, val_size):
    """Sample images for training and validation."""
    print(f"\nSampling {sample_size} images ({val_size} val, {sample_size - val_size} train)...")

    # Get all image IDs
    images = data['images']
    image_ids = [img['id'] for img in images]

    # Shuffle and sample
    random.seed(42)  # For reproducibility
    random.shuffle(image_ids)

    val_ids = set(image_ids[:val_size])
    train_ids = set(image_ids[val_size:sample_size])

    print(f"  Selected {len(train_ids)} training images")
    print(f"  Selected {len(val_ids)} validation images")

    return train_ids, val_ids


def create_subset_annotations(data, image_ids):
    """Create subset annotations for selected images."""
    image_ids_set = set(image_ids)

    # Filter images
    subset_images = [img for img in data['images'] if img['id'] in image_ids_set]

    # Filter annotations
    subset_annotations = [ann for ann in data['annotations'] if ann['image_id'] in image_ids_set]

    # Create image_id to image mapping
    image_map = {img['id']: img for img in subset_images}

    return subset_images, subset_annotations, image_map


def download_image(url, dest_path, retries=3):
    """Download a single image from COCO server."""
    for attempt in range(retries):
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            urllib.request.urlretrieve(url, dest_path)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
                continue
            else:
                return False
    return False


def download_images(image_list, save_dir, max_download=None):
    """Download COCO images from server."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if max_download:
        image_list = image_list[:max_download]

    print(f"\nDownloading {len(image_list)} images to {save_dir}...")

    success = 0
    failed = 0

    for img_info in tqdm(image_list):
        file_name = img_info['file_name']
        dest_path = save_dir / file_name

        if dest_path.exists():
            success += 1
            continue

        # Try train2017 first
        url = f"http://images.cocodataset.org/train2017/{file_name}"

        if download_image(url, dest_path):
            success += 1
        else:
            # Try val2017
            url = f"http://images.cocodataset.org/val2017/{file_name}"
            if download_image(url, dest_path):
                success += 1
            else:
                failed += 1

    print(f"  Downloaded: {success}, Failed: {failed}")
    return success


def convert_bbox_to_yolo(bbox, image_width, image_height):
    """Convert COCO bbox (x, y, w, h) to YOLO format (x_center, y_center, w, h) normalized."""
    x, y, w, h = bbox

    # Calculate center coordinates
    x_center = x + w / 2
    y_center = y + h / 2

    # Normalize to [0, 1]
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    w_norm = w / image_width
    h_norm = h / image_height

    return x_center_norm, y_center_norm, w_norm, h_norm


def create_yolo_labels(annotations, image_map, labels_dir):
    """Create YOLO format label files."""
    print(f"\nCreating YOLO labels in {labels_dir}...")

    # Group annotations by image_id
    image_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Create labels for each image
    created = 0
    for img_id, anns in tqdm(image_annotations.items()):
        if img_id not in image_map:
            continue
        image_info = image_map[img_id]
        img_width = image_info['width']
        img_height = image_info['height']

        # Get file name without extension
        file_name = Path(image_info['file_name']).stem

        # Create label file
        label_path = labels_dir / f"{file_name}.txt"

        with open(label_path, 'w') as f:
            for ann in anns:
                category_id = ann['category_id'] - 1  # COCO uses 1-indexed, YOLO uses 0-indexed
                bbox = ann['bbox']

                # Convert to YOLO format
                x_center, y_center, w, h = convert_bbox_to_yolo(bbox, img_width, img_height)

                # Write line: class x_center y_center width height
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        created += 1

    print(f"  Created {created} label files")


def create_dataset_structure():
    """Create the dataset folder structure."""
    print("\nCreating dataset folder structure...")

    # Create directories
    (DATASET_DIR / "images" / "train2017").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "images" / "val2017").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / "train2017").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / "val2017").mkdir(parents=True, exist_ok=True)

    print(f"  Created structure in {DATASET_DIR}")
    return {
        'train_images': DATASET_DIR / "images" / "train2017",
        'val_images': DATASET_DIR / "images" / "val2017",
        'train_labels': DATASET_DIR / "labels" / "train2017",
        'val_labels': DATASET_DIR / "labels" / "val2017",
    }


def create_yaml_config():
    """Create the YAML configuration file for YOLO training."""
    print("\nCreating YAML config file...")

    yaml_content = f"""# COCO5000 Dataset Configuration
# 5K subset of COCO for YOLO training
# Training: {SAMPLE_SIZE - VAL_SIZE} images
# Validation: {VAL_SIZE} images

path: {DATASET_DIR}
train: images/train2017
val: images/val2017

nc: 80
names:
  - person
  - bicycle
  - car
  - motorcycle
  - airplane
  - bus
  - train
  - truck
  - boat
  - traffic light
  - fire hydrant
  - stop sign
  - parking meter
  - bench
  - bird
  - cat
  - dog
  - horse
  - sheep
  - cow
  - elephant
  - bear
  - zebra
  - giraffe
  - backpack
  - umbrella
  - handbag
  - tie
  - suitcase
  - frisbee
  - skis
  - snowboard
  - sports ball
  - kite
  - baseball bat
  - baseball glove
  - skateboard
  - surfboard
  - tennis racket
  - bottle
  - wine glass
  - cup
  - fork
  - knife
  - spoon
  - bowl
  - banana
  - apple
  - sandwich
  - orange
  - broccoli
  - carrot
  - hot dog
  - pizza
  - donut
  - cake
  - chair
  - couch
  - potted plant
  - bed
  - dining table
  - toilet
  - tv
  - laptop
  - mouse
  - remote
  - keyboard
  - cell phone
  - microwave
  - oven
  - toaster
  - sink
  - refrigerator
  - book
  - clock
  - vase
  - scissors
  - teddy bear
  - hair drier
  - toothbrush
"""

    yaml_path = PROJECT_DIR / "coco5000.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"  Created: {yaml_path}")
    return yaml_path


def main():
    """Main function to create COCO5K dataset."""
    print("=" * 60)
    print("COCO5000 Dataset Preparation")
    print("=" * 60)

    # Check if annotations file exists
    if not ANNOTATIONS_FILE.exists():
        print(f"ERROR: Annotations file not found: {ANNOTATIONS_FILE}")
        return

    # Load annotations
    data = load_annotations(ANNOTATIONS_FILE)

    # Sample images
    train_ids, val_ids = sample_images(data, SAMPLE_SIZE, VAL_SIZE)

    # Create dataset structure
    dirs = create_dataset_structure()

    # Get image info for training and validation
    train_images, train_annotations, train_image_map = create_subset_annotations(data, train_ids)
    val_images, val_annotations, val_image_map = create_subset_annotations(data, val_ids)

    # Download images from COCO server
    print("\nDownloading images from COCO server...")
    print("This may take a while depending on your connection speed.")

    # Download training images (we'll download as many as we can in reasonable time)
    # Start with downloading a subset to test
    downloaded_train = download_images(train_images, dirs['train_images'])
    downloaded_val = download_images(val_images, dirs['val_images'])

    # Create YOLO labels
    create_yolo_labels(train_annotations, train_image_map, dirs['train_labels'])
    create_yolo_labels(val_annotations, val_image_map, dirs['val_labels'])

    # Create YAML config
    yaml_path = create_yaml_config()

    # Count actual images
    train_img_count = len(list(dirs['train_images'].glob('*.jpg'))) + len(list(dirs['train_images'].glob('*.png')))
    val_img_count = len(list(dirs['val_images'].glob('*.jpg'))) + len(list(dirs['val_images'].glob('*.png')))

    print("\n" + "=" * 60)
    print("COCO5000 Dataset Ready!")
    print("=" * 60)
    print(f"Dataset location: {DATASET_DIR}")
    print(f"Config file: {yaml_path}")
    print(f"Training images available: {train_img_count}")
    print(f"Validation images available: {val_img_count}")
    print("\nTo train:")
    print(f"  python train_cht_model.py --data coco5000.yaml --epochs 50 --device cuda:0")


if __name__ == "__main__":
    main()
