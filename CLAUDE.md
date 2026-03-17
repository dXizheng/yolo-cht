# Project: YOLO QAT (Quantization-Aware Training)

## Goal
Build a YOLO model with QAT (Quantization-Aware Training) to introduce sparsity and quantization. Start from pretrained weights.

## Key Details
- **Framework**: YOLO (likely Ultralytics)
- **Approach**: QAT for model compression via sparsity and quantization
- **Pretrained weights**: Use pretrained weights to initialize training
- **Loss function**: Can use either reproduced version or Ultralytics
- **Evaluation metric**: mAP (mean Average Precision)
- **Expected baseline**: With pretrained weights, mAP should be approximately **0.5** initially

## Notes
- QAT enables the model to learn quantization-friendly representations during training
- This allows for efficient deployment with quantized weights (e.g., INT8)
- Sparsity can further reduce model size and inference time
- **Attention layers (softmax) must use FP16 precision** - int8/fp8 is too aggressive for softmax and causes accuracy degradation
  - FP16AttentionWrapper wraps attention modules to run in FP16
  - Conv2d layers inside attention modules are NOT replaced with QATConv2d_CHT (keep original FP32 Conv2d which can handle FP16 input)

## Optimization Approaches for Better Results

### 1. Optimize the Evolution (CHT Sparsity)
- Tune the hyperparameters used in the evolution logic
- Optimize the evolution logic itself
- This affects how the model learns sparsity during training

### 2. Increase Sparsity
- Replace more layers with CHT layers
- Use other methods for sparsity (e.g., sparse attention)
- Higher sparsity leads to more model compression

### 3. Quantization Choice
- Goal: Apply int8/fp8 or even 4-bit quantization if possible
- If some modules/operations are corrupted by quantization (e.g., softmax), use higher precision (BF16)
- Try to use the least amount of higher precision as possible (prefer int8/fp8 over bf16/fp16)
- Current implementation already wraps attention modules with BF16 wrapper for softmax operations

## Coding Guidelines
- Avoid and detect try-except error handling - minimize overly broad exception handling and ensure errors are properly detected rather than silently caught
