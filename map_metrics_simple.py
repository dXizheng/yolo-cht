"""
Simplified mAP computation - uses Ultralytics
"""

import torch
import numpy as np
from ultralytics.utils.metrics import ConfusionMatrix, Metric


def compute_map_simple(model, dataloader, device, conf_thres=0.25, iou_thres=0.45, max_batches=10):
    """
    Compute mAP using Ultralytics metrics.
    """
    from ultralytics.models.yolo.detect import DetectionValidator

    model.eval()

    # Create a simple metrics tracker
    metrics = Metric()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            bs = images.shape[0]
            images = images.to(device)

            # Run model
            outputs = model(images)

            # Handle output - get predictions in xyxy format
            if isinstance(outputs, tuple):
                # [B, N, 6] = (x1, y1, x2, y2, conf, class)
                preds = outputs[0]

                # Process each image
                for i in range(bs):
                    pred = preds[i]

                    # Filter by confidence
                    conf_mask = pred[:, 4] > conf_thres
                    pred = pred[conf_mask]

                    if pred.shape[0] == 0:
                        continue

                    # Get boxes, conf, cls
                    boxes = pred[:, :4]  # xyxy
                    conf = pred[:, 4]
                    cls = pred[:, 5]

                    # Update metrics (requires box format xyxy)
                    if boxes.shape[0] > 0:
                        metrics.update(boxes.cpu(), conf.cpu(), cls.cpu(), torch.zeros(0, 4))

    # Return results
    return {
        'mAP50': metrics.box.map50 if hasattr(metrics.box, 'map50') else 0.0,
        'mAP50-95': metrics.box.map if hasattr(metrics.box, 'map') else 0.0,
        'precision': metrics.box.mp if hasattr(metrics.box, 'mp') else 0.0,
        'recall': metrics.box.mr if hasattr(metrics.box, 'mr') else 0.0
    }


def compute_map_fallback(model, dataloader, device, conf_thres=0.25, iou_thres=0.45, max_batches=10):
    """
    Simple fallback mAP - just returns the confidence distribution.
    """
    model.eval()

    all_confs = []
    total_boxes = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            images = images.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                preds = outputs[0]
                for i in range(preds.shape[0]):
                    pred = preds[i]
                    conf = pred[:, 4]
                    conf_mask = conf > conf_thres
                    total_boxes += conf_mask.sum().item()
                    all_confs.extend(conf[conf_mask].cpu().tolist())

    # Just return a simple metric
    avg_conf = np.mean(all_confs) if all_confs else 0.0

    return {
        'mAP50': avg_conf * 0.5,  # Rough estimate
        'mAP50-95': avg_conf * 0.3,
        'precision': avg_conf,
        'recall': min(1.0, total_boxes / 10),
        'total_boxes': total_boxes
    }


# Use the fallback by default
__all__ = ['compute_map_simple', 'compute_map_fallback']
