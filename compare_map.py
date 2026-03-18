"""
Compare mAP computation between our custom implementation and Ultralytics' built-in val().

This script helps diagnose whether the low mAP issue is due to:
1. Our mAP computation implementation
2. CHT/QAT modifications to the model
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from metrics import compute_ultralytics_map
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import yaml


def create_dataloader(data_yaml, img_size=640, batch_size=16, augment=False):
    """Create a simple dataloader for validation."""
    # Load data config
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    data_root = Path(data['path'])

    class SimpleYOLODataset(Dataset):
        def __init__(self, img_dir, img_size=640):
            self.img_dir = Path(img_dir)
            self.img_size = img_size
            self.img_files = sorted(list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png')))
            self.label_files = [self.img_dir.parent.parent / 'labels' / self.img_dir.name / f'{img.stem}.txt' for img in self.img_files]

        def __len__(self):
            return len(self.img_files)

        def __getitem__(self, idx):
            # Load image
            img = Image.open(self.img_files[idx]).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

            # Load labels - return normalized xywh (batch_idx added by collate_fn)
            label_path = self.label_files[idx]
            labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            labels.append([cls, cx, cy, w, h])

            if len(labels) == 0:
                labels = torch.zeros((0, 5))
            else:
                labels = torch.tensor(labels)

            return img, labels

    def yolo_collate_fn(batch):
        """Collate function that adds batch indices to targets."""
        images = []
        targets = []
        for i, (img, labels) in enumerate(batch):
            images.append(img)
            if labels.numel() > 0:
                # Add batch index as first column
                batch_idx = torch.full((labels.shape[0], 1), i, dtype=labels.dtype)
                labels_with_idx = torch.cat([batch_idx, labels], dim=1)
                targets.append(labels_with_idx)

        images = torch.stack(images)
        if len(targets) > 0:
            targets = torch.cat(targets, dim=0)
        else:
            targets = torch.zeros((0, 6))

        return images, targets

    # Use validation images
    val_img_dir = data_root / data['val']
    dataset = SimpleYOLODataset(val_img_dir, img_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=yolo_collate_fn
    )

    return dataloader


def run_comparison():
    """Run comparison between our mAP and Ultralytics' mAP."""
    print("=" * 70)
    print("mAP Comparison: Custom Implementation vs Ultralytics Built-in")
    print("=" * 70)

    # Configuration
    model_path = "yolov8n.pt"
    data_yaml = "coco128.yaml"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    img_size = 640

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_yaml}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")

    # Create dataloader
    print(f"\nLoading data from {data_yaml}...")
    dataloader = create_dataloader(data_yaml, img_size=img_size, batch_size=batch_size)
    print(f"  Dataset size: {len(dataloader.dataset)} images")

    # Test with Ultralytics built-in val() first
    print("\n" + "=" * 70)
    print("Method 1: Ultralytics built-in val()")
    print("=" * 70)

    model_yolo = YOLO(model_path)
    ultralytics_results = model_yolo.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        verbose=False
    )

    ultralytics_map50 = getattr(ultralytics_results.box, 'map50', None)
    ultralytics_map50_95 = getattr(ultralytics_results.box, 'map', None)

    print(f"  mAP@50:    {ultralytics_map50:.4f}" if ultralytics_map50 else "  mAP@50:    N/A")
    print(f"  mAP@50:95: {ultralytics_map50_95:.4f}" if ultralytics_map50_95 else "  mAP@50:95: N/A")

    # For our custom implementation, use the updated compute_ultralytics_map with Ultralytics NMS
    print("\n" + "=" * 70)
    print("Method 2: Our custom mAP using Ultralytics NMS")
    print("=" * 70)

    # Load model for prediction - use the raw model for getting raw outputs
    model = YOLO(model_path)

    # Get the raw PyTorch model (without Ultralytics wrapper)
    raw_model = model.model
    # Move model to the correct device
    raw_model = raw_model.to(device)

    # Run our custom mAP computation using raw model
    custom_map = compute_ultralytics_map(
        raw_model,
        dataloader,
        device=device,
        conf_threshold=0.001,
        debug=True
    )

    custom_map50 = custom_map.get('mAP50', 0.0)
    custom_map50_95 = custom_map.get('mAP50-95', 0.0)

    print(f"  mAP@50:    {custom_map50:.4f}")
    print(f"  mAP@50:95: {custom_map50_95:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<15} {'Ultralytics':<15} {'Custom':<15}")
    print("-" * 45)
    print(f"{'mAP@50':<15} {ultralytics_map50:<15.4f} {custom_map50:<15.4f}")
    print(f"{'mAP@50:95':<15} {ultralytics_map50_95:<15.4f} {custom_map50_95:<15.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Check if custom mAP matches Ultralytics
    map50_diff = abs(ultralytics_map50 - custom_map50)
    if map50_diff < 0.05:
        print(f"  SUCCESS! Custom mAP@50 ({custom_map50:.4f}) matches Ultralytics ({ultralytics_map50:.4f})")
        print("  The mAP computation is now correct.")
        print("  If YOLO26n still has low mAP, the issue is in the model modifications.")
    else:
        print(f"  Custom mAP@50 ({custom_map50:.4f}) differs from Ultralytics ({ultralytics_map50:.4f})")
        print("  Difference: {:.4f}. Need to debug the NMS integration.".format(map50_diff))

    print()


def main():
    run_comparison()


if __name__ == "__main__":
    main()
