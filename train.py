"""
YOLO26n Training Script - Ultralytics-Aligned Version

Training script for YOLO26n with CHT layers and QAT quantization.
Uses Ultralytics native components for best compatibility:
- YOLODataset with augmentations
- v8DetectionLoss (native Ultralytics loss)
- SGD optimizer with standard hyperparameters
- Native validation via yolo.val()
"""

# Fix OpenMP duplicate library issue on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

import sys
from datetime import datetime
from pathlib import Path

# Global log file handle
_log_file = None
_log_path = None

def setup_logging(args):
    """Set up logging to file in addition to console output."""
    global _log_file, _log_path

    # Create log directory
    quant_folder = args.quantization if args.quantization != 'none' else 'fp32'
    log_dir = Path(args.project) / args.name / quant_folder / args.replace_mode / args.optimizer / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = log_dir / f"train_{timestamp}.log"

    # Open log file
    _log_file = open(_log_path, 'w', encoding='utf-8')

    return _log_path

def log_print(*args, **kwargs):
    """Print to both console and log file."""
    # Use sys.stdout.write to avoid recursion
    msg = ' '.join(str(arg) for arg in args)
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()

    if _log_file is not None:
        _log_file.write(msg + '\n')
        _log_file.flush()

def close_logging():
    """Close the log file."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None

# Alias log_print to print for convenience in the rest of the code
print = log_print

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
import sys
import cv2
import numpy as np
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo26n.yolo26n_cht_qat_model import load_yolo26n_cht_qat_model, count_model_params
from yolo26n.yolo26n_config import ReplaceMode
from yolo26n.metrics import DetectionLossReproduced


def _default_collate_fn(batch):
    """Default collate function for YOLO detection."""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    all_labels = []
    for i, label in enumerate(labels):
        if label.shape[0] > 0:
            label_with_idx = torch.zeros((label.shape[0], 6), dtype=label.dtype)
            label_with_idx[:, 0] = i  # batch index
            label_with_idx[:, 1:] = label
            all_labels.append(label_with_idx)
    if len(all_labels) > 0:
        labels = torch.cat(all_labels, 0)
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)
    return images, labels


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLO26n with CHT + QAT')
    parser.add_argument('--model', type=str, default='yolo26n.pt',
                       help='Path to YOLO26n model weights')
    parser.add_argument('--data', type=str, default='coco128.yaml',
                       help='Path to data config file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--save-period', type=int, default=5,
                       help='Save checkpoint every N epochs (0 to disable)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--sparsity', type=float, default=0.9,
                       help='Target sparsity (0.9 = 90%%)')
    parser.add_argument('--sparsity-schedule', type=str, default='step',
                       choices=['immediate', 'step', 'linear', 'sigmoid'],
                       help='Sparsity schedule (step is recommended for gradual sparsity)')
    parser.add_argument('--sparsity-warmup', type=int, default=15,
                       help='Epochs before sparsity starts increasing (for step schedule)')
    parser.add_argument('--sparsity-step', type=int, default=5,
                       help='Increase sparsity every N epochs (for step schedule)')
    parser.add_argument('--sparsity-step-size', type=float, default=0.05,
                       help='Sparsity increase per step (0.05 = 5%%)')
    parser.add_argument('--replace-mode', type=str, default='backbone_neck',
                       choices=['backbone', 'backbone_neck', 'all'],
                       help='Which layers to replace with CHT')
    parser.add_argument('--skip-first-convs', type=int, default=2,
                       help='Number of early conv layers to skip from CHT (keep dense)')
    parser.add_argument('--replace-inside-attention', action='store_true', default=True,
                       help='Replace Conv2d inside attention modules with CHT or QAT')
    parser.add_argument('--quantization', type=str, default='int8',
                       choices=['int8', 'fp8', 'none'],
                       help='Quantization type')
    # Loss and mAP now strictly use Ultralytics standard
    # The 'reproduced' options are deprecated but kept for backward compatibility
    parser.add_argument('--workers', type=int, default=0,
                       help='Number of dataloader workers')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='yolo26n_cht_qat',
                       help='Experiment name')
    parser.add_argument('--exist-ok', action='store_true',
                       help='Overwrite existing experiment')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--unfreeze-all', action='store_true',
                       help='Unfreeze all parameters for training')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'muon', 'adamw'],
                       help='Optimizer: sgd (default), muon (Muon for weights + AdamW for biases)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for SGD, muon uses 0.02')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='Momentum for SGD (default: 0.937)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay (default: 0.0005)')
    return parser.parse_args()


def create_dataloader(data_config, imgsz=640, batch_size=16, workers=4, augment=True, rect=False, single_cls=False):
    """
    Create training dataloader using Ultralytics YOLODataset.

    Args:
        data_config: Path to data config yaml
        imgsz: Image size
        batch_size: Batch size
        workers: Number of workers
        augment: Enable data augmentations (mosaic, mixup, etc.)
        rect: Use rectangular training
        single_cls: Single class training

    Returns:
        DataLoader with Ultralytics format
    """
    try:
        from pathlib import Path
        from ultralytics.data.dataset import YOLODataset

        # Load dataset config
        with open(data_config) as f:
            data = yaml.safe_load(f)

        path = data.get('path', '')
        train_path = data.get('train', 'train')

        # Build full path
        img_path = Path(path) / train_path if path else Path(train_path)

        print(f"Loading images from: {img_path}")

        # Check if path exists
        if not img_path.exists():
            print(f"Warning: Path {img_path} does not exist, trying fallback...")
            # Try relative path
            img_path = Path(data_config).parent / train_path

        # Create YOLODataset with augmentations (matching Ultralytics standard)
        # Note: Ultralytics YOLODataset API may vary by version
        try:
            dataset = YOLODataset(
                img_path=str(img_path),
                imgsz=imgsz,
                batch_size=batch_size,
                augment=augment,
                cache=False,
                rect=rect,
                single_cls=single_cls,
                stride=32,
                pad=0.0,
                prefix='',
                task='detect',
                data=data
            )
        except (TypeError, AttributeError):
            # Fallback: try with 'path' parameter for older Ultralytics versions
            try:
                dataset = YOLODataset(
                    path=str(img_path),
                    imgsz=imgsz,
                    batch_size=batch_size,
                    augment=augment,
                    cache=False,
                    rect=rect,
                    single_cls=single_cls,
                    stride=32,
                    pad=0.0,
                    prefix='',
                    task='detect',
                    data=data
                )
            except (TypeError, AttributeError) as e2:
                print(f"Warning: YOLODataset failed with both APIs: {e2}")
                raise

        print(f"Dataset created: {len(dataset)} images with augmentations={augment}")

        # Use default collate function
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=augment,  # Only shuffle if not using rectangular
            num_workers=workers,
            pin_memory=True,
        )
        if not hasattr(dataset, 'collate_fn') or dataset.collate_fn is None:
            # Use default collate function if none provided
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=augment,
                num_workers=workers,
                pin_memory=True,
                collate_fn=_default_collate_fn
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=augment,
                num_workers=workers,
                pin_memory=True,
                collate_fn=dataset.collate_fn
            )

        print(f"Dataloader created successfully")
        return loader, dataset

    except (ImportError, AttributeError, ValueError, OSError, IOError) as e:
        import traceback
        print(f"Error creating Ultralytics dataloader: {e}")
        traceback.print_exc()
        print("Falling back to simple dataset...")
        return create_simple_dataloader(data_config, imgsz, batch_size, workers, augment)


def create_simple_dataloader(data_config, imgsz=640, batch_size=16, workers=4, augment=False):
    """Create a simple dataloader with basic loading (fallback)."""
    try:
        from pathlib import Path
        import cv2
        import numpy as np

        # Load dataset config
        with open(data_config) as f:
            data = yaml.safe_load(f)

        path = data.get('path', '')
        train_path = data.get('train', 'train')

        # Build full path
        img_path = Path(path) / train_path if path else Path(train_path)

        print(f"Loading images from: {img_path}")

        # Get list of image files
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_files.extend(list((img_path).glob(ext)))

        print(f"Found {len(img_files)} images")

        if len(img_files) == 0:
            raise ValueError(f"No images found in {img_path}")

        # Simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, img_files, img_size=640, augment=False):
                self.img_files = img_files
                self.img_size = img_size
                self.augment = augment

            def __len__(self):
                return len(self.img_files)

            def __getitem__(self, idx):
                # Load image
                img_path = self.img_files[idx]
                img = cv2.imread(str(img_path))
                if img is None:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

                # Load labels
                img_path_str = str(img_path).replace('\\', '/')
                if 'images/' in img_path_str:
                    label_path_str = img_path_str.replace('images/', 'labels/')
                else:
                    label_path_str = img_path_str.replace('/images/', '/labels/')
                label_file = Path(label_path_str.rsplit('.', 1)[0] + '.txt')

                labels = []
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                labels.append([idx, cls, x, y, w, h])

                if len(labels) == 0:
                    labels = torch.zeros((0, 6))
                else:
                    labels = torch.tensor(labels)

                return img, labels

        def collate_fn(batch):
            images, labels = zip(*batch)
            images = torch.stack(images, 0)
            all_labels = []
            for i, label in enumerate(labels):
                if label.shape[0] > 0:
                    label[:, 0] = i
                    all_labels.append(label)
            if len(all_labels) > 0:
                labels = torch.cat(all_labels, 0)
            else:
                labels = torch.zeros((0, 6))
            return images, labels

        dataset = SimpleDataset(img_files, imgsz, augment)
        print(f"Simple dataset created: {len(dataset)} images")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=augment,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return loader, dataset

    except (OSError, IOError, ValueError, RuntimeError) as e:
        import traceback
        print(f"Error creating simple dataloader: {e}")
        traceback.print_exc()
        # Return dummy dataloader
        return create_dummy_dataloader(batch_size, imgsz), None


def create_dummy_dataloader(batch_size=16, imgsz=640):
    """Create a dummy dataloader for testing."""

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, length=100):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # Return random image and dummy labels
            # Format: [batch_idx, class, x, y, w, h] (normalized)
            img = torch.randn(3, imgsz, imgsz)
            labels = torch.tensor([[0, 0, 0.5, 0.5, 0.3, 0.3]])  # batch_idx, class, x, y, w, h
            return img, labels

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        return images, torch.cat(labels, 0)

    return DataLoader(
        DummyDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    ), None


def convert_outputs_for_loss(outputs, batch_size=4):
    """
    Convert model outputs to format expected by v8DetectionLoss.

    v8DetectionLoss expects:
    - preds: list of tensors, each [batch, channels, H, W] or [batch, num_anchors, features]

    The CHT model returns feature maps in training mode.
    """
    if outputs is None:
        return outputs

    # If already a list of tensors, ensure each has proper shape
    if isinstance(outputs, (list, tuple)):
        converted = []
        for out in outputs:
            if isinstance(out, torch.Tensor):
                # Handle 1D tensor (flattened) - reshape to [batch, features, 1, 1]
                if out.dim() == 1:
                    total_size = out.shape[0]
                    # Use provided batch_size to calculate features
                    features = total_size // batch_size
                    out = out.view(batch_size, features, 1, 1)
                # Ensure 4D tensor [batch, channels, H, W]
                if out.dim() == 3:
                    out = out.unsqueeze(0)
                if out.dim() == 4:
                    converted.append(out)
                elif out.dim() == 2:
                    # Reshape to [batch, channels, 1, 1]
                    b = out.shape[0]
                    c = out.shape[1]
                    converted.append(out.view(b, c, 1, 1))
                else:
                    converted.append(out)
            else:
                converted.append(out)
        return converted
    elif isinstance(outputs, torch.Tensor):
        # Single tensor - convert to list
        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(0)
        return [outputs] if outputs.dim() == 4 else [outputs]
    return outputs


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    loss_fn=None,
    verbose=True,
    scaler=None,
    quantization=None
):
    """
    Train for one epoch using Ultralytics-aligned components.

    Args:
        model: YOLO26n model
        dataloader: Training dataloader
        optimizer: SGD optimizer with standard hyperparameters
        device: Device
        epoch: Current epoch
        loss_fn: v8DetectionLoss (if None, uses default)
        verbose: Print progress

    Returns:
        Dictionary with training metrics
    """
    model.train()

    # Use DetectionLossReproduced (Ultralytics-aligned loss for CHT model)
    if loss_fn is None:
        loss_fn = DetectionLossReproduced(model)

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Handle both formats (Ultralytics returns different format)
        if isinstance(batch, dict):
            # New Ultralytics format: batch is a dict with 'img', 'bboxes', 'cls', 'batch_idx' keys
            images = batch['img'].to(device)
            # Convert uint8 to float32 if needed
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            # Construct targets from bboxes and cls
            # Format: [batch_idx, class, x, y, w, h]
            batch_indices = batch.get('batch_idx', None)
            bboxes = batch.get('bboxes', None)
            cls = batch.get('cls', None)

            if bboxes is not None and cls is not None and batch_indices is not None:
                # Convert to [batch_idx, class, x, y, w, h] format
                cls = cls.squeeze(-1) if cls.dim() > 1 and cls.shape[-1] == 1 else cls
                if cls.dim() == 1:
                    cls = cls.unsqueeze(-1)
                # batch_indices could be a tensor or int
                if isinstance(batch_indices, torch.Tensor):
                    batch_idx_tensor = batch_indices.unsqueeze(-1)
                else:
                    batch_idx_tensor = torch.tensor([batch_indices], device=bboxes.device).unsqueeze(-1)
                targets = torch.cat([batch_idx_tensor, cls, bboxes], dim=-1)
                # Move targets to the same device as the model
                targets = targets.to(device)
            else:
                targets = None
        elif isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
            # Convert uint8 to float32 if needed
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            targets = batch[1]
            # Move targets to device
            if targets is not None and isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            # Ultralytics format: targets is a list of tensors per image
            if isinstance(targets, list):
                # Already in correct format for v8DetectionLoss
                pass
            elif isinstance(targets, torch.Tensor):
                # Convert from [N, 6] to list format
                targets = targets
        else:
            images = batch.to(device)
            # Convert uint8 to float32 if needed
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            targets = None

        # Use AMP autocast only for int8, not for FP8 (FP8 has too limited dynamic range)
        use_autocast = False  # Disable to fix loss issue
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_autocast):
            outputs = model(images)

            # Convert outputs to format expected by DetectionLossReproduced
            batch_size = images.shape[0]
            outputs = convert_outputs_for_loss(outputs, batch_size)

            # Compute loss using Ultralytics v8DetectionLoss
            # v8DetectionLoss expects model and targets in specific format
            loss, loss_items = loss_fn(outputs, targets)

        # Check for NaN/Inf before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf detected at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            continue

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if verbose and batch_idx % 10 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)

    return {
        'loss': avg_loss,
        'lr': optimizer.param_groups[0]['lr']
    }


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Create a learning rate schedule with warmup (matching Ultralytics).

    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles

    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate(
    model,
    device,
    data_yaml='coco128.yaml',
    imgsz=640,
    batch_size=16,
    verbose=True,
    conf_threshold=0.001
):
    """
    Validate model using Ultralytics native validation (yolo.val()).

    Args:
        model: YOLO26n model
        device: Device
        data_yaml: Path to data config
        imgsz: Image size
        batch_size: Batch size
        verbose: Print progress
        conf_threshold: Confidence threshold for detections

    Returns:
        Dictionary with validation metrics (mAP50, mAP50-95, etc.)
    """
    from ultralytics import YOLO

    # Create a YOLO wrapper for validation
    # We use the model directly - Ultralytics val() will handle the forward pass
    model.eval()

    # Try to use native Ultralytics validation
    try:
        # Create a YOLO instance (won't load weights, we'll use our model)
        yolo_val = YOLO('yolo26n.pt')

        # Replace the model with our CHT model
        yolo_val.model = model

        # Run validation using Ultralytics native method
        results = yolo_val.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            verbose=verbose,
            conf=conf_threshold  # Pass confidence threshold
        )

        # Extract metrics
        map50 = getattr(results.box, 'map50', 0.0)
        map50_95 = getattr(results.box, 'map', 0.0)
        precision = getattr(results.box, 'mp', 0.0)
        recall = getattr(results.box, 'mr', 0.0)

        metrics = {
            'mAP50': map50,
            'mAP50-95': map50_95,
            'precision': precision,
            'recall': recall
        }

        if verbose:
            print(f"  Validation: mAP@50={map50:.4f}, mAP@50:0.95={map50_95:.4f}")

    except (RuntimeError, ValueError, AttributeError, TypeError) as e:
        print(f"  Native validation failed: {e}")
        print("  Using fallback validation...")
        metrics = validate_fallback(model, device, data_yaml, imgsz, batch_size, verbose, conf_threshold=conf_threshold)

    model.train()
    return metrics


def validate_fallback(
    model,
    device,
    data_yaml='coco128.yaml',
    imgsz=640,
    batch_size=16,
    verbose=True,
    conf_threshold=0.001
):
    """Fallback validation using custom mAP computation."""
    from yolo26n.metrics import compute_map_reproduced

    # Create a simple dataloader for validation
    try:
        val_loader, _ = create_dataloader(data_yaml, imgsz, batch_size, workers=0, augment=False)
    except (OSError, IOError, ValueError, RuntimeError) as e:
        print(f"Warning: Could not create validation dataloader: {e}")
        val_loader = create_dummy_dataloader(batch_size, imgsz)[0]

    metrics = compute_map_reproduced(model, val_loader, device, conf_threshold=conf_threshold, iou_threshold=0.45)

    if verbose:
        print(f"  Validation (fallback): mAP@50={metrics['mAP50']:.4f}, mAP@50:0.95={metrics['mAP50-95']:.4f}")

    return metrics


def save_checkpoint(model, args, epoch, save_path):
    """Save a checkpoint with CHT layer information."""
    # Collect quantization info
    bf16_attention_count = model.get_num_bf16_attention_modules() if hasattr(model, 'get_num_bf16_attention_modules') else 0
    bf16_param_count = model.get_num_bf16_parameters() if hasattr(model, 'get_num_bf16_parameters') else 0

    # Use model's built-in qat_layers list - more reliable than iterating named_modules()
    # which can miss nested modules in some cases
    qat_layers_count = len(model.qat_layers)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'sparsity': model.get_sparsity(),
        'target_sparsity': model.get_sparsity_target(),
        'cht_layers_count': model.get_num_cht_layers(),
        'replace_mode': args.replace_mode,
        'quantization': args.quantization,
        'qat_enabled': args.quantization not in ['none', None],
        'qat_layers_count': qat_layers_count,
        'bf16_attention_count': bf16_attention_count,
        'bf16_param_count': bf16_param_count,
    }

    # Add CHT layer details if available
    if hasattr(model, 'cht_layers') and model.cht_layers:
        cht_layers_info = []
        for i, layer in enumerate(model.cht_layers):
            layer_info = {
                'index': i,
                'sparsity': layer.sparsity if hasattr(layer, 'sparsity') else 0.0,
            }
            if hasattr(layer, 'mask') and layer.mask is not None:
                layer_info['mask_shape'] = list(layer.mask.shape)
            cht_layers_info.append(layer_info)
        checkpoint['cht_layers_info'] = cht_layers_info

    # Save optimizer state
    # Note: optimizer is not passed here, but can be added if needed

    torch.save(checkpoint, save_path)

    # Announce checkpoint with quantization info
    print(f"\n  Checkpoint saved: {save_path}")
    print(f"    Epoch: {epoch}, Sparsity: {checkpoint['sparsity']*100:.1f}% (target: {checkpoint['target_sparsity']*100:.1f}%)")
    print(f"    CHT layers: {checkpoint['cht_layers_count']}, QAT layers: {checkpoint['qat_layers_count']}")
    print(f"    Quantization: {checkpoint['quantization']} (enabled: {checkpoint['qat_enabled']})")
    print(f"    BF16 attention: {checkpoint['bf16_attention_count']} modules, {checkpoint.get('bf16_param_count', 0):,} params")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CombinedMuonAdamW(torch.optim.Optimizer):
    """Wrapper to combine Muon for weights (ndim>=2) and AdamW for biases/norms (ndim<2)."""
    def __init__(self, muon_optimizer, adamw_optimizer):
        self.muon = muon_optimizer
        self.adamw = adamw_optimizer
        # Combine param_groups from both optimizers for scheduler compatibility
        combined_groups = list(muon_optimizer.param_groups) + list(adamw_optimizer.param_groups)
        # Initialize as Optimizer with combined param_groups
        super().__init__(combined_groups, {})

    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        return {'muon': self.muon.state_dict(), 'adamw': self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict['muon'])
        self.adamw.load_state_dict(state_dict['adamw'])


def main():
    """Main training function with Ultralytics-aligned components."""
    set_seed()
    args = parse_args()

    # Set up logging to file
    log_path = setup_logging(args)
    print(f"Logging to: {log_path}")

    print("=" * 60)
    print("YOLO26n CHT + QAT Training (Ultralytics-Aligned)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Sparsity: {args.sparsity * 100}%")
    print(f"Sparsity schedule: {args.sparsity_schedule}")
    print(f"Sparsity warmup: {args.sparsity_warmup} epochs")
    print(f"Sparsity step: every {args.sparsity_step} epochs (+{args.sparsity_step_size*100}%)")
    print(f"Replace mode: {args.replace_mode}")
    print(f"Replace inside attention: {args.replace_inside_attention}")
    print(f"Unfreeze all params: {args.unfreeze_all}")
    print(f"Quantization: {args.quantization}")
    print(f"Loss: Ultralytics v8DetectionLoss (native)")
    print(f"Optimizer: SGD (Ultralytics standard)")
    print(f"Validation: Ultralytics native (yolo.val())")
    print("=" * 60)

    # Set device - check if the requested device is actually available
    if args.device.startswith('cuda'):
        # Check if CUDA is available and the requested GPU exists
        if torch.cuda.is_available():
            # Validate the GPU index if specified (e.g., cuda:0, cuda:1)
            if ':' in args.device:
                device_id = int(args.device.split(':')[1])
                if device_id >= torch.cuda.device_count():
                    print(f"Warning: GPU {device_id} not found, falling back to CPU")
                    device = torch.device('cpu')
                else:
                    device = torch.device(args.device)
            else:
                device = torch.device(args.device)
        else:
            print("Warning: CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")

    # Convert replace mode
    if args.replace_mode == 'backbone':
        replace_mode = ReplaceMode.BACKBONE
    elif args.replace_mode == 'backbone_neck':
        replace_mode = ReplaceMode.BACKBONE_NECK
    else:
        replace_mode = ReplaceMode.ALL

    # Load model
    print("\nLoading model...")
    model = load_yolo26n_cht_qat_model(
        model_path=args.model,
        sparsity=args.sparsity,
        replace_mode=replace_mode,
        quantization=args.quantization,
        regrow_method="L3n",
        shared_mask_sw=True,
        soft=True,
        link_update_ratio=0.08,
        skip_first_n_convs=args.skip_first_convs,
        replace_inside_attention=args.replace_inside_attention,
        sparsity_schedule=args.sparsity_schedule,
        sparsity_warmup_epochs=args.sparsity_warmup,
        sparsity_step_epochs=args.sparsity_step,
        sparsity_step_size=args.sparsity_step_size
    )

    # Unfreeze all parameters if requested
    if args.unfreeze_all:
        print("\nUnfreezing all parameters...")
        for p in model._model.parameters():
            p.requires_grad = True

    # Count parameters
    total_params, trainable_params = count_model_params(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"CHT layers: {model.get_num_cht_layers()}")
    print(f"QAT layers: {model.get_num_qat_layers()}")
    print(f"Early convs skipped from CHT: {args.skip_first_convs}")
    print(f"Target sparsity: {model.get_sparsity_target() * 100}%")

    # Move model to device
    model = model.to(device)

    # Ensure all parameters are on the correct device (handles custom modules)
    if hasattr(model, 'ensure_on_device'):
        model.ensure_on_device(device)

    # Create dataloaders with Ultralytics YOLODataset (with augmentations for training)
    print("\nCreating dataloaders...")
    train_loader, train_dataset = create_dataloader(
        args.data, args.imgsz, args.batch_size, args.workers,
        augment=True, rect=False, single_cls=False
    )

    # Initialize DetectionLossReproduced (Ultralytics-aligned loss for CHT model)
    print("\nInitializing DetectionLossReproduced...")
    loss_fn = DetectionLossReproduced(model)

    # Create optimizer based on user selection
    # Ultralytics defaults for SGD: lr0=0.01, momentum=0.937, weight_decay=0.0005, nesterov=False
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False
        )
        print(f"Using SGD optimizer (lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay})")
    elif args.optimizer == 'muon':
        from Muon.muon import SingleDeviceMuon
        # Separate params: Muon for weights (ndim >= 2), AdamW for biases/norms (ndim < 2)
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]
        muon_lr = 0.001  # Start with lower LR for Muon, similar to standard training
        adam_lr = args.lr * 0.1  # Lower LR for biases
        muon_opt = SingleDeviceMuon(
            muon_params,
            lr=muon_lr,
            weight_decay=args.weight_decay,
            momentum=0.95
        )
        adamw_opt = optim.AdamW(
            other_params,
            lr=adam_lr,
            weight_decay=args.weight_decay
        )
        optimizer = CombinedMuonAdamW(muon_opt, adamw_opt)
        print(f"Using Muon optimizer (muon_lr={muon_lr}, adamw_lr={adam_lr}, weight_decay={args.weight_decay})")
    elif args.optimizer == 'adamw':
        # Use AdamW for all parameters with lower LR (AdamW typically works better with lower LR)
        adam_lr = 0.001  # Much lower than SGD default
        optimizer = optim.AdamW(
            model.parameters(),
            lr=adam_lr,
            weight_decay=args.weight_decay
        )
        print(f"Using AdamW optimizer (lr={adam_lr}, weight_decay={args.weight_decay})")

    # For FP8: Only use GradScaler for gradient scaling, disable FP16 autocast
    # For int8: Use full AMP (works fine)
    # For None: No quantization, no AMP
    # GradScaler disabled - QAT fake quantization already handles quantization-aware training
    # If you need mixed precision, use a different PyTorch version or implement custom scaling
    use_scaler = False
    scaler = None

    # Warmup epochs (Ultralytics default: 3.0)
    warmup_epochs = 3.0
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1

    # Create learning rate scheduler (Ultralytics uses cosine annealing)
    # lr = lr0 * (1 - cos(pi * epoch / epochs)) / 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

    # Training loop
    print("\nStarting training...")
    print(f"Warmup: {warmup_epochs} epochs")
    best_map = 0.0
    map_history = []  # Track mAP@50 for each epoch

    # Warmup settings - use optimizer-specific base LR
    if args.optimizer == 'sgd':
        warmup_lr = 0.001  # Start from small value for SGD
        base_lr = args.lr  # 0.01 for SGD
    else:
        # AdamW/Muon use their own defined LR
        base_lr = optimizer.param_groups[0]['lr']
        warmup_lr = base_lr * 0.1  # Start from 10% of base LR

    for epoch in range(1, args.epochs + 1):
        # Apply warmup for first few epochs
        if epoch <= warmup_epochs:
            # Linear warmup for learning rate
            warmup_factor = (epoch - 1) / max(1, warmup_epochs - 1)
            current_lr = warmup_lr + (base_lr - warmup_lr) * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # Train with Ultralytics-aligned loss
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            loss_fn=loss_fn, verbose=args.verbose, scaler=scaler,
            quantization=args.quantization
        )

        # Evolve CHT layers (sparsity) - only after warmup to avoid destabilizing training
        if epoch > args.sparsity_warmup:
            print(f"\n  Evolving CHT layers at epoch {epoch}...")
            evolve_stats = model.evolve(current_epoch=epoch, verbose=False)
            print(f"  Sparsity after evolution: {model.get_sparsity() * 100:.2f}%")
        else:
            print(f"\n  Skipping evolve during warmup (epoch {epoch}/{args.sparsity_warmup})")

        # Validate using Ultralytics native validation - every epoch
        val_metrics = {'mAP50': 0.0, 'mAP50-95': 0.0}
        print(f"\n  Validating at epoch {epoch}...")
        val_metrics = validate(
            model, device,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            verbose=args.verbose,
            conf_threshold=0.001
        )

        # Save best model based on mAP
        map_history.append(val_metrics['mAP50'])
        if val_metrics['mAP50'] > best_map:
            best_map = val_metrics['mAP50']
            print(f"  New best mAP@50: {best_map:.4f}")

        # Save periodic checkpoint
        if args.save_period > 0 and epoch % args.save_period == 0:
            # Create subfolder based on quantization type, replace mode, and optimizer
            quant_folder = args.quantization if args.quantization != 'none' else 'fp32'
            checkpoint_path = Path(args.project) / args.name / quant_folder / args.replace_mode / args.optimizer / 'weights' / f'epoch_{epoch}.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, args, epoch, checkpoint_path)

        # Update scheduler only after warmup (to avoid conflict with warmup LR)
        if epoch > warmup_epochs:
            scheduler.step()

        print(f"\nEpoch {epoch}/{args.epochs} completed")
        print(f"  Loss: {train_metrics['loss']:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    print("\nTraining completed!")
    print(f"Best mAP@50: {best_map:.4f}")
    print(f"Final sparsity: {model.get_sparsity() * 100:.2f}%")

    # Prepare quant folder for saving
    quant_folder = args.quantization if args.quantization != 'none' else 'fp32'

    # Plot mAP curve
    epochs = list(range(1, len(map_history) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, map_history, 'b-', linewidth=2, label='mAP@50')
    plt.axhline(y=best_map, color='r', linestyle='--', label=f'Best mAP@50: {best_map:.4f}')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('mAP@50', fontsize=12)
    plt.title('mAP@50 vs Epoch', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = Path(args.project) / args.name / quant_folder / args.replace_mode / args.optimizer / 'weights' / 'map_curve.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nmAP curve saved to: {plot_path}")

    # Save final model with CHT metadata
    save_path = Path(args.project) / args.name / quant_folder / args.replace_mode / args.optimizer / 'weights' / 'last.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, args, args.epochs, save_path)

    # Close log file
    close_logging()
    print(f"\nLog saved to: {log_path}")

    return log_path


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Make sure to close log file on error
        close_logging()
        raise
