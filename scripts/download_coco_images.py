"""
Quick COCO Image Downloader

This script downloads COCO images from the official server.
"""

import os
import requests
from pathlib import Path
import urllib.request
import ssl

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
COCO_URL = "http://images.cocodataset.org/zips"


def download_file(url, dest_path, retries=3):
    """Download a single file with retries."""
    for attempt in range(retries):
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            urllib.request.urlretrieve(url, dest_path)
            return True
        except Exception as e:
            print(f"  Retry {attempt+1}/{retries}: {e}")
            time.sleep(1)
    return False


def download_images_from_list(image_list, save_dir, max_download=None):
    """Download a list of COCO images."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if max_download:
        image_list = image_list[:max_download]

    print(f"Downloading {len(image_list)} images to {save_dir}...")

    success = 0
    failed = 0

    for img_info in tqdm(image_list):
        file_name = img_info['file_name']
        dest_path = save_dir / file_name

        if dest_path.exists():
            success += 1
            continue

        # Try train2017 first, then val2017
        url = f"http://images.cocodataset.org/train2017/{file_name}"

        if download_file(url, dest_path):
            success += 1
        else:
            # Try val2017
            url = f"http://images.cocodataset.org/val2017/{file_name}"
            if download_file(url, dest_path):
                success += 1
            else:
                failed += 1

    print(f"Downloaded: {success}, Failed: {failed}")
    return success, failed


def download_coco_val2017(dest_dir, max_images=5000):
    """Download COCO val2017 annotations and images."""
    print("Downloading COCO val2017 annotations...")
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    dest_dir = Path(dest_dir)
    zip_path = dest_dir / "annotations_trainval2017.zip"

    if not zip_path.exists():
        print(f"  Downloading {annotations_url}...")
        download_file(annotations_url, zip_path)

    # Extract if needed
    extract_dir = dest_dir / "annotations"
    if not extract_dir.exists():
        print("  Extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest_dir)

    # Load annotations
    val_annotations = extract_dir / "annotations" / "instances_val2017.json"
    if val_annotations.exists():
        print(f"  Loading {val_annotations}...")
        import json
        with open(val_annotations, 'r') as f:
            data = json.load(f)
        return data['images']
    return None


def download_sample_train_images(dest_dir, sample_count=4500):
    """Download a sample of COCO train2017 images."""
    print(f"Downloading {sample_count} COCO train2017 images...")

    # COCO train2017 has ~118K images, we need to download a subset
    # We'll try to download by checking which images are in the annotations

    # For now, let's create a simple download - this may take a while
    # In practice, you'd want to download the full zip or use a mirror

    print("Note: Downloading full COCO train2017 (~18GB) is not practical.")
    print("Please consider:")
    print("  1. Downloading COCO val2017 separately")
    print("  2. Using a local COCO dataset if available")
    print("  3. Downloading train2017 from a mirror")

    return []


if __name__ == "__main__":
    import json

    # Test download with a few images
    annotations_file = PROJECT_DIR / "instances_minitrain2017.json"

    if annotations_file.exists():
        print("Loading annotations...")
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        # Get sample images
        sample_images = data['images'][:100]  # First 100 for testing

        # Download to test folder
        test_dir = PROJECT_DIR / "coco5000" / "test_download"
        download_images_from_list(sample_images, test_dir)
