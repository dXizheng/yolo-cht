"""
Test script to validate int8 and fp8 quantization for training and validation.
"""

import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from yolo26n_cht_qat_model import load_yolo26n_cht_qat_model
from yolo26n_config import ReplaceMode, create_cht_config


def test_quantization_mode(quantization: str, sparsity: float = 0.0):
    """Test a quantization mode with training and validation."""
    print(f"\n{'='*60}")
    print(f"TESTING QUANTIZATION: {quantization.upper()}")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model with specified quantization
    print(f"\n[1/4] Loading model with {quantization} quantization...")
    try:
        model = load_yolo26n_cht_qat_model(
            model_path="yolo26n.pt",
            sparsity=sparsity,
            replace_mode=ReplaceMode.BACKBONE_NECK,
            quantization=quantization,
            regrow_method="L3n",
            shared_mask_sw=True,
            soft=True,
            link_update_ratio=0.1,
            skip_first_n_convs=2,
            replace_inside_attention=True,
            sparsity_schedule='immediate'
        )
        model = model.to(device)
        model.train()
        print(f"    Model loaded successfully!")
    except Exception as e:
        print(f"    ERROR loading model: {e}")
        return False

    # Test forward pass
    print(f"\n[2/4] Testing forward pass...")
    try:
        model.train()
        test_input = torch.randn(1, 3, 640, 640).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"    Forward pass successful! Output shape: {output[0].shape}")
    except Exception as e:
        print(f"    ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test training step (one batch)
    print(f"\n[3/4] Testing training step (backward pass)...")
    try:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Create dummy batch
        images = torch.randn(2, 3, 640, 640).to(device)
        # Create dummy targets in YOLO format
        # Format: [batch_idx, class_id, x, y, w, h] normalized
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],  # batch 0
            [1, 1, 0.3, 0.7, 0.1, 0.15], # batch 1
        ]).to(device)

        # Forward pass
        outputs = model(images)

        # Simple loss - just use the output
        loss = sum(o.abs().mean() for o in outputs if o is not None)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"    Training step successful! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"    ERROR in training step: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test validation (inference mode)
    print(f"\n[4/4] Testing validation (inference)...")
    try:
        model.eval()
        val_input = torch.randn(1, 3, 640, 640).to(device)
        with torch.no_grad():
            val_output = model(val_input)
        print(f"    Validation successful! Output shape: {val_output[0].shape}")
    except Exception as e:
        print(f"    ERROR in validation: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'='*60}")
    print(f"RESULT: {quantization.upper()} - ALL TESTS PASSED!")
    print(f"{'='*60}\n")
    return True


def test_all_quantization_modes():
    """Test all quantization modes: int8, fp8, and none."""
    results = {}

    # Test int8
    results['int8'] = test_quantization_mode('int8', sparsity=0.0)

    # Test fp8
    results['fp8'] = test_quantization_mode('fp8', sparsity=0.0)

    # Test none (FP32 baseline)
    results['none'] = test_quantization_mode('none', sparsity=0.0)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for mode, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {mode.upper()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\nAll quantization modes work correctly!")
    else:
        print("\nSome quantization modes failed!")

    return all_passed


if __name__ == "__main__":
    success = test_all_quantization_modes()
    sys.exit(0 if success else 1)
