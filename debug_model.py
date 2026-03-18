#!/usr/bin/env python3
"""Debug script to check BF16 wrapper and model setup."""

import torch
from yolo26n_cht_qat_model import load_yolo26n_cht_qat_model

print("Loading model...")
model = load_yolo26n_cht_qat_model(
    'yolo26n.pt',
    sparsity=0.9,
    quantization='int8',
    replace_inside_attention=True
)

print(f"\n=== Model Info ===")
print(f"CHT layers: {model.get_num_cht_layers()}")
print(f"QAT layers: {model.get_num_qat_layers()}")
print(f"BF16 attention modules: {len(model.bf16_attention_modules)}")

# Check BF16 wrapper attributes
if model.bf16_attention_modules:
    wrapper = model.bf16_attention_modules[0]
    print(f"\n=== BF16 Wrapper Check ===")
    print(f"wrapper.i: {wrapper.i}")  # Should NOT be -1
    print(f"wrapper.save: {wrapper.save}")  # Should NOT be empty
    print(f"wrapped_module is child: {'wrapped_module' in dict(wrapper.named_children())}")

    # Check training mode propagation
    model = model.cuda()
    model.train()
    print(f"\n=== Training Mode ===")
    print(f"model.training: {model.training}")
    print(f"model._model.training: {model._model.training}")
    print(f"wrapper.training: {wrapper.training}")
    print(f"wrapper.wrapped_module.training: {wrapper.wrapped_module.training}")

# Test forward pass
print(f"\n=== Forward Pass Test ===")
model.eval()
x = torch.randn(1, 3, 640, 640).cuda()
with torch.no_grad():
    out = model(x)

print(f"Eval output type: {type(out)}")
if isinstance(out, tuple):
    print(f"  Length: {len(out)}")
    if len(out) >= 1:
        print(f"  [0] shape: {out[0].shape if hasattr(out[0], 'shape') else 'N/A'}")
elif isinstance(out, list):
    print(f"  Length: {len(out)}")

model.train()
x = torch.randn(1, 3, 640, 640).cuda()
out = model(x)

print(f"\nTrain output type: {type(out)}")
if isinstance(out, (list, tuple)):
    print(f"  Length: {len(out)}")
    for i, o in enumerate(out):
        if hasattr(o, 'shape'):
            print(f"  [{i}] shape: {o.shape}")
            print(f"  [{i}] mean: {o.mean().item():.4f}, std: {o.std().item():.4f}")
