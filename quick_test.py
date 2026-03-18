"""
Quick verification test
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo26n.yolo26n_cht_qat_model import load_yolo26n_cht_qat_model
from yolo26n.yolo26n_config import ReplaceMode
from ultralytics import YOLO


def test():
    print("="*60)
    print("QUICK VERIFICATION TEST")
    print("="*60)

    # Load models
    standard_model = YOLO("yolo26n.pt").model
    standard_model.eval()

    cht_model = load_yolo26n_cht_qat_model(
        model_path="yolo26n.pt",
        sparsity=0.0,
        replace_mode=ReplaceMode.BACKBONE_NECK,
        quantization='none',
        regrow_method="L3n",
        shared_mask_sw=True,
        soft=True,
        link_update_ratio=0.1,
        sparsity_schedule='immediate'
    )
    cht_model.eval()

    # Test input
    test_input = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        std_out = standard_model(test_input)
        cht_out = cht_model(test_input)

    diff = (std_out[0] - cht_out[0]).abs()

    print(f"\nOutput comparison:")
    print(f"  Standard: {std_out[0].shape}")
    print(f"  CHT: {cht_out[0].shape}")
    print(f"  Max diff: {diff.max().item():.10f}")
    print(f"  Mean diff: {diff.mean().item():.10f}")

    if diff.max().item() < 0.001:
        print("\n*** SUCCESS! CHT model outputs match standard model! ***")
    else:
        print("\n*** ISSUE: Outputs don't match! ***")


if __name__ == "__main__":
    test()
