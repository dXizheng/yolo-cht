"""
Compare mAP between Standard YOLO26n and CHT YOLO model
"""

import os
import torch
import sys
import argparse
from pathlib import Path

# Add current directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from yolo26n_cht_qat_model import load_yolo26n_cht_qat_model
from yolo26n_config import ReplaceMode


def run_comparison(args):
    print("=" * 70)
    print("mAP COMPARISON: Standard YOLO26n vs CHT YOLO")
    print("=" * 70)

    # Use specified device or auto-detect
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Validate GPU exists before using
    if device and 'cuda' in device:
        cuda_idx = int(device.split(':')[1]) if ':' in device else 0
        if cuda_idx >= torch.cuda.device_count():
            cuda_idx = 0
            device = f"cuda:{cuda_idx}"
            print(f"Requested GPU not available, using {device}")

    print(f"\nDevice: {device}")

    # Configuration - use relative paths for Linux compatibility
    data_yaml = str(project_root / "coco128.yaml")
    model_path = str(project_root / "yolo26n.pt")
    img_size = 640
    batch_size = 16

    # ============================================================
    # Standard YOLO26n Model
    # ============================================================
    print("\n" + "=" * 70)
    print("Loading Standard YOLO26n...")
    print("=" * 70)

    try:
        standard_model = YOLO(model_path)
        print("Running validation on standard model...")
        standard_results = standard_model.val(
            data=data_yaml,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            verbose=True
        )

        std_map50 = getattr(standard_results.box, 'map50', None)
        std_map50_95 = getattr(standard_results.box, 'map', None)
        std_precision = getattr(standard_results.box, 'mp', None)
        std_recall = getattr(standard_results.box, 'mr', None)

        print(f"\n  Standard YOLO26n Results:")
        print(f"    mAP@50:    {std_map50:.4f}" if std_map50 else "    mAP@50:    N/A")
        print(f"    mAP@50:95: {std_map50_95:.4f}" if std_map50_95 else "    mAP@50:95: N/A")
        print(f"    Precision: {std_precision:.4f}" if std_precision else "    Precision: N/A")
        print(f"    Recall:    {std_recall:.4f}" if std_recall else "    Recall:    N/A")

    except Exception as e:
        print(f"Error with standard model: {e}")
        import traceback
        traceback.print_exc()
        std_map50 = std_map50_95 = std_precision = std_recall = None

    # ============================================================
    # CHT YOLO Model (0% sparsity - baseline)
    # ============================================================
    print("\n" + "=" * 70)
    print("Loading CHT YOLO Model (0% sparsity - baseline)...")
    print("=" * 70)

    try:
        cht_model = load_yolo26n_cht_qat_model(
            model_path=model_path,
            sparsity=0.0,  # Start with 0 sparsity to compare baseline
            replace_mode=ReplaceMode.BACKBONE_NECK,
            quantization='none',
            regrow_method="L3n",
            shared_mask_sw=True,
            soft=True,
            link_update_ratio=0.1,
            sparsity_schedule='immediate'
        )
        cht_model = cht_model.to(device)
        cht_model.eval()

        # Wrap in YOLO for validation
        cht_yolo = YOLO(model_path)
        cht_yolo.model = cht_model

        print("Running validation on CHT model...")
        cht_results = cht_yolo.val(
            data=data_yaml,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            verbose=True
        )

        cht_map50 = getattr(cht_results.box, 'map50', None)
        cht_map50_95 = getattr(cht_results.box, 'map', None)
        cht_precision = getattr(cht_results.box, 'mp', None)
        cht_recall = getattr(cht_results.box, 'mr', None)

        print(f"\n  CHT YOLO Results (0% sparsity):")
        print(f"    mAP@50:    {cht_map50:.4f}" if cht_map50 else "    mAP@50:    N/A")
        print(f"    mAP@50:95: {cht_map50_95:.4f}" if cht_map50_95 else "    mAP@50:95: N/A")
        print(f"    Precision: {cht_precision:.4f}" if cht_precision else "    Precision: N/A")
        print(f"    Recall:    {cht_recall:.4f}" if cht_recall else "    Recall:    N/A")

    except Exception as e:
        print(f"Error with CHT model: {e}")
        import traceback
        traceback.print_exc()
        cht_map50 = cht_map50_95 = cht_precision = cht_recall = None

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if std_map50 is not None and cht_map50 is not None:
        print(f"\n{'Metric':<15} {'Standard':<15} {'CHT (0% spar)':<15} {'Difference':<15}")
        print("-" * 60)

        diff_map50 = cht_map50 - std_map50
        diff_map50_95 = cht_map50_95 - std_map50_95 if (std_map50_95 and cht_map50_95) else 0
        diff_prec = cht_precision - std_precision if (std_precision and cht_precision) else 0
        diff_rec = cht_recall - std_recall if (std_recall and cht_recall) else 0

        print(f"{'mAP@50':<15} {std_map50:<15.4f} {cht_map50:<15.4f} {diff_map50:+.4f}")
        print(f"{'mAP@50:95':<15} {std_map50_95:<15.4f} {cht_map50_95:<15.4f} {diff_map50_95:+.4f}")
        print(f"{'Precision':<15} {std_precision:<15.4f} {cht_precision:<15.4f} {diff_prec:+.4f}")
        print(f"{'Recall':<15} {std_recall:<15.4f} {cht_recall:<15.4f} {diff_rec:+.4f}")

        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)

        if abs(diff_map50) < 0.01:
            print("  CHT model (0% sparsity) matches standard model!")
            print("  The CHT modifications don't affect baseline accuracy.")
        else:
            print(f"  CHT model shows {diff_map50:+.4f} mAP@50 difference from standard.")
            if diff_map50 < 0:
                print("  WARNING: CHT modifications are degrading performance even at 0% sparsity!")
            else:
                print("  Unexpected: CHT is improving performance?")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare mAP between Standard YOLO26n and CHT YOLO model")
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda", "cuda:0", "cpu"). If not specified, auto-detect.')
    args = parser.parse_args()
    run_comparison(args)
