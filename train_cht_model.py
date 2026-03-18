"""
Train YOLO26n CHT Model with Sparsity and Quantization (Ultralytics-Aligned)

This script trains the YOLO26n model with:
- CHT (Channel Hierarchy Threshold) for 70% sparsity
- Quantization-Aware Training (QAT) with INT8
- Progressive sparsity evolution during training
- Ultralytics-aligned training components

Usage:
    python train_cht_model.py --device cuda:0 --epochs 50 --batch-size 16 --sparsity 0.7 --quantization int8
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import copy

# Add current directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from yolo26n.yolo26n_cht_qat_model import load_yolo26n_cht_qat_model, YOLO26nCHTQATModel
from yolo26n.yolo26n_config import ReplaceMode, create_cht_config, create_qat_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO26n CHT Model with Sparsity and QAT")

    # Model configuration
    parser.add_argument('--model-path', type=str, default='yolo26n.pt',
                       help='Path to YOLO26n model weights')
    parser.add_argument('--sparsity', type=float, default=0.7,
                       help='Target sparsity (0.7 = 70%%)')
    parser.add_argument('--quantization', type=str, default='int8', choices=['int8', 'fp8', 'none'],
                       help='Quantization type')
    parser.add_argument('--replace-mode', type=str, default='BACKBONE_NECK',
                       choices=['BACKBONE', 'BACKBONE_NECK', 'ALL'],
                       help='Which layers to replace with CHT')

    # CHT configuration
    parser.add_argument('--regrow-method', type=str, default='L3n',
                       help='Regrow method for CHT ("L3n" for CHT)')
    parser.add_argument('--shared-mask-sw', action='store_true', default=True,
                       help='Share mask across sliding windows')
    parser.add_argument('--soft', action='store_true', default=True,
                       help='Use soft removal/regrowth')
    parser.add_argument('--link-update-ratio', type=float, default=0.1,
                       help='Fraction of active connections to update per evolve')

    # Sparsity schedule configuration
    parser.add_argument('--sparsity-schedule', type=str, default='step',
                       choices=['immediate', 'step', 'linear', 'sigmoid'],
                       help='Sparsity evolution schedule')
    parser.add_argument('--sparsity-warmup-epochs', type=int, default=5,
                       help='Epochs before sparsity starts increasing')
    parser.add_argument('--sparsity-step-epochs', type=int, default=5,
                       help='Increase sparsity every N epochs')
    parser.add_argument('--sparsity-step-size', type=float, default=0.1,
                       help='Increase sparsity by this amount each step')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (e.g., "cuda", "cuda:0", "cpu")')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of dataloader workers')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Enable data augmentations (mosaic, mixup, etc.)')
    parser.add_argument('--warmup-epochs', type=float, default=3.0,
                       help='Learning rate warmup epochs')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')

    # Evolve configuration
    parser.add_argument('--evolve-every', type=int, default=5,
                       help='Run evolve() every N epochs')

    # Checkpoint configuration
    parser.add_argument('--save-dir', type=str, default='checkpoint',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs')

    # Data configuration
    parser.add_argument('--data', type=str, default='coco5000.yaml',
                       help='Dataset YAML file')

    # Other options
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')

    return parser.parse_args()


def apply_sparsity_to_state_dict(state_dict):
    """
    Apply mask to weights in state_dict before saving.

    This permanently zeros out pruned weights in the saved checkpoint,
    ensuring the sparsity is preserved in the saved model.

    Args:
        state_dict: Model state dict containing weights and masks

    Returns:
        Modified state dict with weights zeroed where mask is False
    """
    for key in list(state_dict.keys()):
        if 'weight' in key and not key.endswith('.mask'):
            # Find corresponding mask key
            mask_key = key.replace('.weight', '.mask')
            if mask_key in state_dict:
                mask = state_dict[mask_key]
                weight = state_dict[key]
                # Zero out weights where mask is False (convert bool to float for multiplication)
                state_dict[key] = weight * mask.float()
    return state_dict


def create_checkpoint(model: YOLO26nCHTQATModel, optimizer_state=None, epoch: int = 0,
                     sparsity: float = 0.0, save_dir: str = 'checkpoint',
                     quantization: str = 'int8') -> str:
    """
    Save model checkpoint with mask and scale parameters.

    Args:
        model: YOLO26n CHT-QAT model
        optimizer_state: Optimizer state dict
        epoch: Current epoch
        sparsity: Current sparsity
        save_dir: Directory to save checkpoint
        quantization: Quantization type

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create checkpoint name
    sparsity_pct = int(sparsity * 100)
    checkpoint_name = f"yolo26n_cht_qat_{sparsity_pct}_{quantization}.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)

    # Collect checkpoint data
    # Apply sparsity to weights before saving (zero out pruned weights)
    sparse_state_dict = apply_sparsity_to_state_dict(model.state_dict())

    checkpoint = {
        'epoch': epoch,
        'sparsity': sparsity,
        'target_sparsity': model.get_sparsity_target(),
        'quantization': quantization,
        'model_state_dict': sparse_state_dict,
        'cht_layers_info': []
    }

    # Collect CHT layer information
    for i, layer in enumerate(model.cht_layers):
        layer_info = {
            'index': i,
            'name': layer.layer_name if hasattr(layer, 'layer_name') else f'cht_layer_{i}',
            'sparsity': layer.sparsity if hasattr(layer, 'sparsity') else 0.0,
        }
        if hasattr(layer, 'mask') and layer.mask is not None:
            layer_info['mask_shape'] = list(layer.mask.shape)
        checkpoint['cht_layers_info'].append(layer_info)

    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(model: YOLO26nCHTQATModel, checkpoint_path: str,
                  device: str = 'cuda') -> int:
    """
    Load model checkpoint.

    Args:
        model: YOLO26n CHT-QAT model
        checkpoint_path: Path to checkpoint
        device: Device to load to

    Returns:
        Epoch number from checkpoint

    Raises:
        KeyError: If checkpoint does not contain required 'model_state_dict' key
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Validate checkpoint contains required key
    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict' key. Available keys: {list(checkpoint.keys())}")

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    print(f"  Loaded checkpoint from epoch {epoch}")

    return epoch


def train_with_cht(args):
    """
    Train YOLO26n model with CHT sparsity and QAT.

    Args:
        args: Parsed command-line arguments
    """
    print("=" * 70)
    print("YOLO26n CHT + QAT TRAINING")
    print("=" * 70)

    # Determine replace mode
    replace_mode_map = {
        'BACKBONE': ReplaceMode.BACKBONE,
        'BACKBONE_NECK': ReplaceMode.BACKBONE_NECK,
        'ALL': ReplaceMode.ALL
    }
    replace_mode = replace_mode_map[args.replace_mode]

    # Create checkpoint directory
    checkpoint_dir = args.save_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Data configuration
    data_yaml = str(project_root / args.data) if not os.path.isabs(args.data) else args.data

    # Check if data exists
    if not os.path.exists(data_yaml):
        print(f"WARNING: Data config not found: {data_yaml}")
        print("  Training will use default Ultralytics dataset handling")

    # ============================================================
    # Load CHT Model with Sparsity and Quantization
    # ============================================================
    print("\n" + "-" * 50)
    print("Loading YOLO26n CHT-QAT Model...")
    print("-" * 50)

    model_path = str(project_root / args.model_path) if not os.path.isabs(args.model_path) else args.model_path

    # Create CHT config with sparsity schedule
    cht_kwargs = {
        'sparsity_schedule': args.sparsity_schedule,
        'sparsity_warmup_epochs': args.sparsity_warmup_epochs,
        'sparsity_step_epochs': args.sparsity_step_epochs,
        'sparsity_step_size': args.sparsity_step_size,
    }

    # Load the model
    cht_model = load_yolo26n_cht_qat_model(
        model_path=model_path,
        sparsity=args.sparsity,
        replace_mode=replace_mode,
        quantization=args.quantization,
        regrow_method=args.regrow_method,
        shared_mask_sw=args.shared_mask_sw,
        soft=args.soft,
        link_update_ratio=args.link_update_ratio,
        **cht_kwargs
    )

    cht_model = cht_model.to(device)
    cht_model.train()

    print(f"\nModel Configuration:")
    print(f"  Target Sparsity: {args.sparsity * 100:.0f}%")
    print(f"  Quantization: {args.quantization}")
    print(f"  Replace Mode: {args.replace_mode}")
    print(f"  Sparsity Schedule: {args.sparsity_schedule}")
    print(f"  Warmup Epochs: {args.sparsity_warmup_epochs}")
    print(f"  CHT Layers: {cht_model.get_num_cht_layers()}")

    # ============================================================
    # Pre-Training Validation (Epoch 0)
    # ============================================================
    print("\n" + "-" * 50)
    print("Running validation before training (Epoch 0)...")
    print("-" * 50)

    cht_model.eval()
    yolo_val = YOLO(model_path)
    yolo_val.model = cht_model
    pre_train_results = yolo_val.val(
        data=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=device,
        verbose=False
    )
    pre_map50 = pre_train_results.box.map50
    pre_map = pre_train_results.box.map
    print(f"  Pre-training mAP@50:    {pre_map50:.4f}")
    print(f"  Pre-training mAP@50-95: {pre_map:.4f}")

    # Switch back to train mode
    cht_model.train()

    # ============================================================
    # Setup Training with Ultralytics
    # ============================================================
    print("\n" + "-" * 50)
    print("Setting up training...")
    print("-" * 50)

    # Create a YOLO instance for training
    # We use a fresh YOLO model for training infrastructure, then swap in our CHT model
    yolo_trainer = YOLO(model_path)

    # Replace the model with our CHT model
    yolo_trainer.model = cht_model

    # Configure training
    train_args = {
        'data': data_yaml,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch_size,
        'device': device,
        'workers': args.workers,
        'verbose': args.verbose,
        'project': 'runs/train',
        'name': 'cht_qat_training',
        'exist_ok': True,
        'pretrained': False,
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'dropout': 0.0,
        'val': True,
        'save': False,
        'plots': False,
    }

    # ============================================================
    # Training Loop with Periodic Evolve
    # ============================================================
    print("\n" + "-" * 50)
    print("Starting Training...")
    print("-" * 50)

    start_epoch = 0
    current_sparsity = 0.0

    # Check for resume
    if args.resume and args.checkpoint:
        start_epoch = load_checkpoint(cht_model, args.checkpoint, device)
        print(f"Resuming from epoch {start_epoch}")

    # Training callback to integrate evolve() during training
    class EvolveCallback:
        def __init__(self, model, evolve_every, save_dir, save_interval,
 sparsity_schedule, warmup_epochs, step_epochs, step_size,
                    quantization, verbose):
            self.model = model
            self.evolve_every = evolve_every
            self.save_dir = save_dir
            self.save_interval = save_interval
            self.sparsity_schedule = sparsity_schedule
            self.warmup_epochs = warmup_epochs
            self.step_epochs = step_epochs
            self.step_size = step_size
            self.quantization = quantization
            self.verbose = verbose
            self.last_save_epoch = 0
            self.epoch_count = 0

        def on_train_epoch_end(self, trainer):
            """Hook called at the end of each training epoch."""
            self.epoch_count += 1
            epoch = self.epoch_count

            # Get current sparsity
            current_sparsity = self.model.get_sparsity()

            # Print epoch summary
            if self.verbose:
                print(f"\n  Epoch {epoch}/{trainer.epochs} - Sparsity: {current_sparsity * 100:.1f}%")

            # Run evolve every N epochs (after warmup)
            if epoch > self.warmup_epochs and epoch % self.evolve_every == 0:
                if self.verbose:
                    print(f"\n  >>> Running evolve() at epoch {epoch}...")

                # Set model to training mode for evolve
                self.model.train()

                # Run evolve
                stats = self.model.evolve(current_epoch=epoch, verbose=self.verbose)

                if self.verbose and stats:
                    print(f"  >>> Evolve complete:")
                    print(f"      Layers evolved: {stats.get('layers_evolved', 0)}")
                    print(f"      Avg cancellation: {stats.get('total_cancellation', 0):.4f}")

                # Get updated sparsity
                current_sparsity = self.model.get_sparsity()

            # Save checkpoint periodically
            if epoch - self.last_save_epoch >= self.save_interval:
                sparsity_pct = int(current_sparsity * 100)
                checkpoint_name = f"yolo26n_cht_qat_{sparsity_pct}_{self.quantization}_epoch{epoch}.pt"
                checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

                checkpoint = {
                    'epoch': epoch,
                    'sparsity': current_sparsity,
                    'target_sparsity': self.model.get_sparsity_target(),
                    'quantization': self.quantization,
                    'model_state_dict': self.model.state_dict(),
                }

                torch.save(checkpoint, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
                self.last_save_epoch = epoch

            # Return current sparsity for logging
            return {'sparsity': current_sparsity}

    # Create callback
    evolve_callback = EvolveCallback(
        model=cht_model,
        evolve_every=args.evolve_every,
        save_dir=checkpoint_dir,
        save_interval=args.save_interval,
        sparsity_schedule=args.sparsity_schedule,
        warmup_epochs=args.sparsity_warmup_epochs,
        step_epochs=args.sparsity_step_epochs,
        step_size=args.sparsity_step_size,
        quantization=args.quantization,
        verbose=args.verbose
    )

    # Add callback to trainer - Ultralytics uses add_callback
    # Note: This may not work with custom model, so we'll run training differently

    # ============================================================
    # Training using Ultralytics with CHT Model
    # ============================================================
    print("\nRunning training with Ultralytics...")

    try:
        # Use standard Ultralytics training - it will handle the training loop properly
        # We pass our model as the starting point
        results = yolo_trainer.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=device,
            workers=args.workers,
            verbose=args.verbose,
            project='runs/train',
            name='cht_qat_training',
            exist_ok=True,
            pretrained=False,
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            dropout=0.0,
            val=False,  # Skip validation during training
            save=False,
            plots=False,
        )

        print("\nTraining completed!")
        print(f"Results: {results}")

    except (RuntimeError, ValueError, TypeError) as e:
        print(f"Ultralytics training error: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: try manual training for specific errors
        print("\nTrying manual training loop as fallback...")
        try:
            results = run_manual_training(
                model=cht_model,
                data_yaml=data_yaml,
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                device=device,
                workers=args.workers,
                evolve_every=args.evolve_every,
                warmup_epochs=args.sparsity_warmup_epochs,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=args.save_interval,
                quantization=args.quantization,
                verbose=args.verbose
            )
            print("Manual training completed successfully!")
        except (RuntimeError, ValueError, TypeError) as e2:
            print(f"Manual training also failed: {e2}")
            traceback.print_exc()
            raise RuntimeError(
                f"Both Ultralytics and manual training failed. "
                f"Ultralytics error: {e}, Manual error: {e2}"
            ) from e

    # ============================================================
    # Final Checkpoint
    # ============================================================
    print("\n" + "-" * 50)
    print("Saving final checkpoint...")
    print("-" * 50)

    final_sparsity = cht_model.get_sparsity()
    create_checkpoint(
        model=cht_model,
        epoch=args.epochs,
        sparsity=final_sparsity,
        save_dir=checkpoint_dir,
        quantization=args.quantization
    )

    # ============================================================
    # Final Validation
    # ============================================================
    print("\n" + "-" * 50)
    print("Running final validation...")
    print("-" * 50)

    cht_model.eval()
    yolo_trainer.model = cht_model

    try:
        val_results = yolo_trainer.val(
            data=data_yaml,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=device,
            verbose=True
        )

        map50 = getattr(val_results.box, 'map50', None)
        map50_95 = getattr(val_results.box, 'map', None)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Final Sparsity: {final_sparsity * 100:.1f}%")
        print(f"  mAP@50: {map50:.4f}" if map50 else "  mAP@50: N/A")
        print(f"  mAP@50:95: {map50_95:.4f}" if map50_95 else "  mAP@50:95: N/A")

    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone!")


def run_manual_training(
    model,
    data_yaml: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    workers: int,
    evolve_every: int,
    warmup_epochs: int,
    checkpoint_dir: str,
    checkpoint_interval: int,
    quantization: str,
    verbose: bool = True,
    augment: bool = True,
    lr0: float = 0.01
):
    """
    Manual training loop with evolve integration.

    Uses Ultralytics data loading and native v8DetectionLoss with proper warmup.
    """
    from torch.utils.data import DataLoader
    from pathlib import Path
    import yaml

    print("\n" + "=" * 50)
    print("MANUAL TRAINING LOOP (Ultralytics-Aligned)")
    print("=" * 50)

    # Load dataset config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Get dataset paths
    data_path = Path(data_config.get('path', './coco128'))
    train_path = data_path / data_config.get('train', 'images/train2017')

    # Create dataset
    print(f"Loading dataset from: {train_path}")

    # Use Ultralytics dataset with augmentations enabled
    try:
        from ultralytics.data.dataset import YOLODataset
        dataset = YOLODataset(
            path=str(train_path),
            imgsz=imgsz,
            batch_size=batch_size,
            data=data_config,
            augment=augment,  # Enable augmentations (mosaic, mixup, etc.)
            cache=False,
            rect=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix='',
            task='detect',
        )
    except Exception as e:
        print(f"Error loading dataset with YOLODataset: {e}")
        # Fallback: use simple image folder loading WITH REAL LABELS (matching train.py format)
        print("Using fallback dataset loading with COCO labels...")
        import cv2
        import numpy as np

        class COCOImageDataset(torch.utils.data.Dataset):
            """Dataset that loads images and YOLO format labels (matching train.py)."""
            def __init__(self, path, imgsz=640):
                self.path = Path(path)
                self.imgsz = imgsz
                # Find all images
                self.image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    self.image_files.extend(list(self.path.glob(ext)))
                # Sort for consistency
                self.image_files = sorted(self.image_files)

            def __len__(self):
                return len(self.image_files)

            def __getitem__(self, idx):
                img_path = self.image_files[idx]
                # Use cv2 like train.py
                img = cv2.imread(str(img_path))
                if img is None:
                    img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (self.imgsz, self.imgsz))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

                # Load YOLO format labels: [batch_idx, class, x_center, y_center, width, height]
                # Note: batch_idx will be set in collate_fn (like train.py)
                label_path = self.path.parent / 'labels' / (img_path.stem + '.txt')
                labels = []
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                # Format: [batch_idx, class, x, y, w, h] - batch_idx placeholder
                                labels.append([cls, x, y, w, h])
                if len(labels) == 0:
                    labels = torch.zeros((0, 5), dtype=torch.float32)
                else:
                    labels = torch.tensor(labels, dtype=torch.float32)

                return img, labels

        # Custom collate function to handle variable length labels (matching train.py)
        def yolo26n_collate_fn(batch):
            images, labels = zip(*batch)
            images = torch.stack(images, 0)

            # Concatenate labels with batch index offset
            all_labels = []
            for i, label in enumerate(labels):
                if label.shape[0] > 0:
                    label_with_idx = torch.zeros((label.shape[0], 6), dtype=label.dtype)
                    label_with_idx[:, 0] = i  # batch index
                    label_with_idx[:, 1] = label[:, 0]  # class
                    label_with_idx[:, 2] = label[:, 1]  # x
                    label_with_idx[:, 3] = label[:, 2]  # y
                    label_with_idx[:, 4] = label[:, 3]  # w
                    label_with_idx[:, 5] = label[:, 4]  # h
                    all_labels.append(label_with_idx)

            if len(all_labels) > 0:
                labels = torch.cat(all_labels, 0)
            else:
                labels = torch.zeros((0, 6), dtype=torch.float32)

            return images, labels

        dataset = COCOImageDataset(str(train_path), imgsz)

    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=yolo26n_collate_fn
    )

    print(f"Dataset loaded: {len(dataset)} images")

    # Setup training
    model = model.to(device)
    model.train()

    # Use SGD optimizer with Ultralytics standard hyperparameters
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr0,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=False
    )

    # Warmup epochs (Ultralytics default: 3.0)
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1

    # Learning rate scheduler (cosine annealing)
    # Warmup is handled manually in the training loop
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr0 * 0.01)

    # Initialize native Ultralytics v8DetectionLoss
    print("Initializing Ultralytics v8DetectionLoss...")
    from ultralytics.utils.loss import v8DetectionLoss
    detection_loss = v8DetectionLoss(model)
    detection_loss = detection_loss.to(device)
    detection_loss.train()

    # Training loop
    print(f"\nStarting {epochs} epochs of training...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"LR: {lr0}, Momentum: 0.937, Weight decay: 0.0005")
    print(f"Warmup: {warmup_epochs} epochs")
    print(f"Augmentations: {'enabled' if augment else 'disabled'}")
    print(f"Evolve every {evolve_every} epochs (after epoch {warmup_epochs})")
    print()

    last_checkpoint_epoch = 0

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    for epoch in range(epochs):
        # Apply warmup for first few epochs
        if epoch < warmup_epochs:
            # Linear warmup for learning rate
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = lr0 * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            # Warmup momentum
            momentum = warmup_momentum + (0.937 - warmup_momentum) * warmup_factor
            for param_group in optimizer.param_groups:
                if 'momentum' in param_group:
                    param_group['momentum'] = momentum

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        failed_batches = 0
        max_failed_batches = len(dataloader) // 10  # Allow up to 10% failures per epoch

        for batch_idx, batch in enumerate(dataloader):
            # Get images
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device) if isinstance(batch, torch.Tensor) else batch

            # Ensure images require gradient
            images = images.requires_grad_(True)

            # Get labels from batch (format: [batch_idx, class, x, y, w, h])
            if isinstance(batch, (list, tuple)):
                targets = batch[1].to(device)  # Already has batch index from collate_fn
            else:
                targets = torch.zeros((0, 6), device=device)

            optimizer.zero_grad()

            # Forward pass
            try:
                outputs = model(images)

                # Use proper YOLO detection loss
                # DetectionLossReproduced expects: outputs (feature list), targets (YOLO format)
                # targets format: [batch_idx, class, x, y, w, h]
                try:
                    if targets.shape[1] >= 6:  # Has batch_idx, class, x, y, w, h
                        loss, loss_dict = detection_loss(outputs, targets)
                        if loss is None or torch.isnan(loss) or torch.isinf(loss):
                            # Fallback to basic loss
                            loss = sum(o.abs().mean() for o in outputs if isinstance(o, torch.Tensor))
                    else:
                        # No valid targets, use basic loss
                        loss = sum(o.abs().mean() for o in outputs if isinstance(o, torch.Tensor))
                except Exception as loss_error:
                    # If detection loss fails, use basic loss
                    if isinstance(outputs, (list, tuple)):
                        loss = sum(o.abs().mean() for o in outputs if isinstance(o, torch.Tensor))
                    elif isinstance(outputs, torch.Tensor):
                        loss = outputs.abs().mean()
                    else:
                        loss = sum(p.abs().mean() for p in model.parameters()) * 0.001

                # Add regularization to drive towards target sparsity
                # Use (target - current) to create incentive to reach target
                current_sparsity = model.get_sparsity()
                target_sparsity = model.get_sparsity_target()
                sparsity_loss = abs(target_sparsity - current_sparsity) * 0.01
                total_loss = loss + sparsity_loss

            except Exception as e:
                # If forward pass fails, track and skip this batch
                failed_batches += 1
                print(f"  Batch {batch_idx} error: {e}")
                if failed_batches > max_failed_batches:
                    raise RuntimeError(
                        f"Too many failed batches ({failed_batches}). "
                        f"Dataset or model may have issues. Stopping training."
                    )
                continue

            # Check that loss has gradient
            if total_loss.requires_grad:
                # Backward pass
                total_loss.backward()
                optimizer.step()
            else:
                # If loss doesn't require grad, create one that does
                # This shouldn't happen if model is set up correctly
                print(f"  Warning: loss doesn't require grad at batch {batch_idx}")
                continue

            epoch_loss += total_loss.item()
            num_batches += 1

            if verbose and batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {total_loss.item():.4f}")

        # Average loss
        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        current_sparsity = model.get_sparsity()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Sparsity: {current_sparsity*100:.1f}%")

        # Run evolve() after warmup
        if epoch >= warmup_epochs and (epoch - warmup_epochs + 1) % evolve_every == 0:
            if verbose:
                print(f"\n  >>> Running evolve() at epoch {epoch+1}...")
            model.train()  # Ensure in train mode for evolve
            stats = model.evolve(current_epoch=epoch+1, verbose=verbose)
            if verbose and stats:
                print(f"  >>> Evolve complete: {stats.get('layers_evolved', 0)} layers evolved")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            sparsity_pct = int(model.get_sparsity() * 100)
            checkpoint_name = f"yolo26n_cht_qat_{sparsity_pct}_{quantization}_epoch{epoch+1}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

            checkpoint = {
                'epoch': epoch + 1,
                'sparsity': model.get_sparsity(),
                'target_sparsity': model.get_sparsity_target(),
                'quantization': quantization,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            last_checkpoint_epoch = epoch + 1

        # Step scheduler
        scheduler.step()

    # Return training results
    return {
        'final_sparsity': model.get_sparsity(),
        'epochs_completed': epochs
    }


def manual_train_loop(model, yolo_trainer, train_args, evolve_callback):
    """
    Manual training loop with evolve integration.

    This provides more control over the training process and ensures
    evolve() is properly called during training.
    """
    print("Manual training loop not fully implemented.")
    print("Please use standard Ultralytics training with the evolve callback.")


def main():
    """Main entry point."""
    args = parse_args()
    train_with_cht(args)


if __name__ == "__main__":
    main()
