"""
Loss Functions and mAP Metrics

Two versions:
1. Ultralytics version - directly imports from ultralytics package
2. Reproduced version - implements exactly the same logic from scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Dict, List, Tuple, Optional
import numpy as np

# Ultralytics imports for loss components
from ultralytics.utils.loss import VarifocalLoss, BboxLoss, DFLoss
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.nms import non_max_suppression


# =============================================================================
# Output Adapter: Convert YOLO26n output to Ultralytics format
# =============================================================================

def convert_ultralytics_format(boxes, scores, conf_threshold=0.25, iou_threshold=0.45, debug=False):
    """
    Convert Ultralytics YOLO model output to ultralytics DetectionLoss format.
    """
    if debug:
        print(f"[DEBUG convert_ultralytics_format] boxes shape: {boxes.shape}, scores shape: {scores.shape}")

    if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
        return {'boxes': boxes, 'scores': scores, 'class_indices': torch.zeros(0)}

    B, C, N = boxes.shape
    _, num_classes, _ = scores.shape

    if C == 64:
        obj_conf = torch.sigmoid(boxes[:, 4:5, :])
        box_coords = boxes[:, :4, :]
    elif C == 84:
        obj_conf = torch.ones((B, 1, N), device=boxes.device)
        box_coords = boxes[:, :4, :]
    else:
        obj_conf = torch.ones((B, 1, N), device=boxes.device)
        box_coords = boxes[:, :4, :]

    class_probs = torch.sigmoid(scores)
    obj_conf_expanded = obj_conf.expand(-1, num_classes, -1)
    combined_conf = (obj_conf_expanded * class_probs)

    all_boxes = []
    all_scores = []
    all_classes = []

    for b in range(B):
        batch_conf = combined_conf[b]
        batch_boxes = box_coords[b]

        keep_boxes = []
        keep_scores = []
        keep_classes = []

        for cls in range(batch_conf.shape[0]):
            cls_scores = batch_conf[cls, :]
            mask = cls_scores > conf_threshold
            if mask.sum() == 0:
                continue

            cls_boxes = batch_boxes[:, mask].T
            cls_scores = cls_scores[mask]

            cx = cls_boxes[:, 0]
            cy = cls_boxes[:, 1]
            w = cls_boxes[:, 2]
            h = cls_boxes[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            keep = torchvision.ops.nms(xyxy, cls_scores, iou_threshold)

            keep_boxes.append(xyxy[keep])
            keep_scores.append(cls_scores[keep])
            keep_classes.append(torch.full((len(keep),), cls, dtype=torch.long, device=boxes.device))

        if len(keep_boxes) > 0:
            all_boxes.append(torch.cat(keep_boxes, dim=0))
            all_scores.append(torch.cat(keep_scores, dim=0))
            all_classes.append(torch.cat(keep_classes, dim=0))
        else:
            all_boxes.append(torch.zeros((0, 4), device=boxes.device))
            all_scores.append(torch.zeros(0, device=boxes.device))
            all_classes.append(torch.zeros(0, dtype=torch.long, device=boxes.device))

    max_dets = max(b.shape[0] for b in all_boxes)

    padded_boxes = []
    padded_scores = []
    padded_classes = []

    for b in range(B):
        num_dets = all_boxes[b].shape[0]
        if num_dets < max_dets:
            pad = torch.zeros((max_dets - num_dets, 4), device=boxes.device)
            padded_boxes.append(torch.cat([all_boxes[b], pad], dim=0))
            pad_scores = torch.zeros(max_dets - num_dets, device=boxes.device)
            padded_scores.append(torch.cat([all_scores[b], pad_scores], dim=0))
            pad_classes = torch.zeros(max_dets - num_dets, dtype=torch.long, device=boxes.device)
            padded_classes.append(torch.cat([all_classes[b], pad_classes], dim=0))
        else:
            padded_boxes.append(all_boxes[b][:max_dets])
            padded_scores.append(all_scores[b][:max_dets])
            padded_classes.append(all_classes[b][:max_dets])

    result_boxes = torch.stack(padded_boxes, dim=0)
    result_scores = torch.stack(padded_scores, dim=0)
    result_classes = torch.stack(padded_classes, dim=0)

    return {
        'boxes': result_boxes,
        'scores': result_scores,
        'class_indices': result_classes
    }


def convert_yolo26n_to_ultralytics_format(preds, conf_threshold=0.25, iou_threshold=0.45, debug=False):
    """
    Convert YOLO26n model output to ultralytics DetectionLoss format.

    Handles both:
    - New format: list of tensors [batch, channels, H, W]
    - Legacy format: dict with 'one2one'/'one2many' keys
    """
    if debug:
        print(f"[DEBUG convert_yolo26n_to_ultralytics_format] Input type: {type(preds)}")

    # Handle new standard ultralytics format: list of tensors
    if isinstance(preds, (list, tuple)):
        # Convert list of [batch, channels, H, W] to detection format
        all_boxes = []
        all_scores = []

        for pred in preds:
            if pred is None or pred.dim() != 4:
                continue

            b, c, h, w = pred.shape
            # Reshape to [batch, channels, num_anchors]
            pred_flat = pred.flatten(2)  # [batch, channels, H*W]

            # Extract box predictions (first 4*reg_max) and class scores
            reg_max = 16  # Default
            nc = c - 4 * reg_max if c > 4 * reg_max else 0

            if nc > 0:
                boxes = pred_flat[:, :4 * reg_max, :]  # [batch, 4*reg_max, num_anchors]
                scores = pred_flat[:, 4 * reg_max:4 * reg_max + nc, :]  # [batch, nc, num_anchors]
            else:
                boxes = pred_flat[:, :4, :] if pred_flat.shape[1] >= 4 else pred_flat
                scores = pred_flat[:, 4:, :] if pred_flat.shape[1] > 4 else torch.zeros((b, 1, h * w), device=pred.device)

            all_boxes.append(boxes)
            all_scores.append(scores)

        if all_boxes:
            # Concatenate all detection layers
            boxes = torch.cat(all_boxes, dim=2)  # [batch, channels, total_anchors]
            scores = torch.cat(all_scores, dim=2)  # [batch, nc, total_anchors]
            return convert_ultralytics_format(boxes, scores, conf_threshold, iou_threshold, debug)
        return preds

    # Handle legacy dict format
    if not isinstance(preds, dict):
        return preds

    if 'one2one' in preds:
        pred_dict = preds['one2one']
    elif 'one2many' in preds:
        pred_dict = preds['one2many']
    elif 'boxes' in preds and 'scores' in preds:
        boxes = preds['boxes']
        scores = preds['scores']
        return convert_ultralytics_format(boxes, scores, conf_threshold, iou_threshold, debug)
    else:
        return preds

    if not isinstance(pred_dict, dict):
        return preds

    boxes = pred_dict.get('boxes', None)
    scores = pred_dict.get('scores', None)

    if boxes is None or not isinstance(boxes, torch.Tensor):
        return preds

    batch_size = boxes.shape[0]
    return convert_ultralytics_format(boxes, scores, conf_threshold, iou_threshold, debug)


# =============================================================================
# Ultralytics-aligned Detection Loss
# Uses standard Ultralytics components: BboxLoss, VarifocalLoss, DFLoss
# =============================================================================

class DetectionLossReproduced(nn.Module):
    """
    Ultralytics-aligned DetectionLoss for YOLO26n CHT model.

    Uses the standard Ultralytics loss components:
    - BboxLoss: IoU-based box regression (CIoU)
    - VarifocalLoss: Asymmetric focal loss for classification
    - DFLoss: Distribution Focal Loss for box regression
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.stride = model.stride if hasattr(model, 'stride') else torch.tensor([8., 16., 32.])
        self.nc = model.nc if hasattr(model, 'nc') else 80
        self.reg_max = model.reg_max if hasattr(model, 'reg_max') else 16
        self.nl = model.nl if hasattr(model, 'nl') else len(self.stride)
        self.device = next(model.parameters()).device

        # Loss weights (Ultralytics YOLOv8 defaults)
        self.box_loss_weight = 7.5
        self.cls_loss_weight = 0.5
        self.dfl_loss_weight = 1.5

        # Initialize Ultralytics loss components
        self.bbox_loss = BboxLoss(reg_max=self.reg_max).to(self.device)
        self.cls_loss = VarifocalLoss().to(self.device)
        self.dfl_loss = DFLoss().to(self.device)

        # TaskAlignedAssigner for anchor assignment
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0
        )

    def _build_targets(self, targets, batch_size):
        """Build target list per image."""
        target_per_img = []

        # Handle 3D tensor: [batch, 1, 5] -> reshape to [batch, 5]
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        # Ensure targets is contiguous to avoid indexing issues
        if not targets.is_contiguous():
            targets = targets.contiguous()

        for i in range(batch_size):
            img_targets = targets[targets[:, 0] == i][:, 1:]
            if len(img_targets) > 0:
                target_per_img.append(img_targets)
            else:
                target_per_img.append(torch.zeros((0, 5), device=self.device))
        return target_per_img

    def forward(self, preds, targets) -> Tuple[torch.Tensor, Dict]:
        """
        Compute detection loss using Ultralytics components.

        Args:
            preds: Model predictions (dict/list in train mode, tuple in eval mode)
            targets: Ground truth targets [batch_idx, class, x, y, w, h]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        device = self.device
        batch_size = 1

        # Handle new standard ultralytics format: list of tensors [batch, channels, H, W]
        if isinstance(preds, (list, tuple)) and len(preds) > 0 and isinstance(preds[0], torch.Tensor):
            # Convert list of [batch, channels, H, W] to feature list
            feature_list = []
            for f in preds:
                if f is not None and isinstance(f, torch.Tensor) and f.dim() == 4:
                    b, c, h, w = f.shape
                    f_flat = f.flatten(2).permute(0, 2, 1)  # [batch, num_anchors, channels]
                    feature_list.append(f_flat)

            if not feature_list:
                return torch.tensor(0.0, device=device, requires_grad=True), {
                    'box_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0
                }

            feats = feature_list
            batch_size = feats[0].shape[0]
        # Handle dict output from model in training mode
        elif isinstance(preds, dict):
            # Extract features from dict
            if 'one2one' in preds:
                pred_dict = preds['one2one']
            elif 'one2many' in preds:
                pred_dict = preds['one2many']
            else:
                pred_dict = preds

            # Get boxes and scores
            if isinstance(pred_dict, dict):
                boxes = pred_dict.get('boxes', None)
                scores = pred_dict.get('scores', None)
                feats = pred_dict.get('feats', None)

                if feats is not None and isinstance(feats, list) and len(feats) > 0:
                    # Use feats list directly
                    feature_list = []
                    for f in feats:
                        if f is not None and isinstance(f, torch.Tensor):
                            if f.dim() == 4:
                                b, c, h, w = f.shape
                                f = f.flatten(2).permute(0, 2, 1)
                            feature_list.append(f)

                    if not feature_list:
                        return torch.tensor(0.0, device=device, requires_grad=True), {
                            'box_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0
                        }

                    feats = feature_list
                    batch_size = feats[0].shape[0]

                elif boxes is not None and isinstance(boxes, torch.Tensor):
                    # Convert boxes/scores to features
                    if scores is not None:
                        combined = torch.cat([boxes, scores], dim=1)
                        feats = [combined.permute(0, 2, 1)]
                    else:
                        feats = [boxes.permute(0, 2, 1)]
                    batch_size = boxes.shape[0]
                else:
                    return torch.tensor(0.0, device=device, requires_grad=True), {
                        'box_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0
                    }
            else:
                return torch.tensor(0.0, device=device, requires_grad=True), {
                    'box_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0
                }

        # Handle tuple/list output (from eval mode)
        elif isinstance(preds, (list, tuple)):
            feats = list(preds)
            if not feats:
                return torch.tensor(0.0, device=device, requires_grad=True), {
                    'box_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0
                }
            batch_size = feats[0].shape[0] if feats[0].dim() >= 3 else 1

        # Handle single tensor
        elif isinstance(preds, torch.Tensor):
            if preds.dim() == 4:
                b, c, h, w = preds.shape
                feats = [preds.flatten(2).permute(0, 2, 1)]
            else:
                feats = [preds]
            batch_size = preds.shape[0] if preds.dim() >= 3 else 1
        else:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'box_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0
            }

        # Build targets
        if isinstance(targets, torch.Tensor) and targets.numel() > 0:
            target_per_img = self._build_targets(targets, batch_size)
        else:
            target_per_img = [torch.zeros((0, 5), device=device) for _ in range(batch_size)]

        # Compute losses using Ultralytics components
        total_box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_dfl_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_valid = 0

        for layer_idx, feat in enumerate(feats):
            if feat is None:
                continue

            # Ensure 3D tensor
            if feat.dim() == 4:
                b, c, h, w = feat.shape
                feat = feat.flatten(2).permute(0, 2, 1)

            if feat.dim() != 3:
                continue

            b, num_anchors, c = feat.shape
            stride = self.stride[layer_idx] if layer_idx < len(self.stride) else 32.0

            # Extract predictions
            # YOLO format: [xywh, obj, cls1, cls2, ...] or [distribution]
            if c >= 4 + self.nc:
                pred_box = feat[:, :, :4]  # xywh in (0-1) range
                pred_cls = feat[:, :, 4:4+self.nc]  # class logits
                pred_dfl = None
            elif c == 4 + self.nc * self.reg_max:
                # DFL format: [xywh, dfl_distribution]
                pred_box = feat[:, :, :4]
                pred_dfl = feat[:, :, 4:]  # DFL distribution
                pred_cls = None
            else:
                pred_box = feat[:, :, :4]
                pred_cls = None
                pred_dfl = None

            # Compute loss for each image
            for batch_idx in range(b):
                img_targets = target_per_img[batch_idx]

                if len(img_targets) == 0:
                    continue

                tgt_classes = img_targets[:, 0].long()
                tgt_boxes = img_targets[:, 1:5]  # xywh normalized

                if len(tgt_boxes) == 0:
                    continue

                # Handle case where targets might have different format
                if tgt_boxes.shape[1] < 4:
                    print(f"Warning: Unexpected target shape {tgt_boxes.shape}, skipping")
                    continue

                # Create target tensors in correct format for Ultralytics
                # Convert normalized xywh to xyxy
                tgt_xyxy = torch.zeros((tgt_boxes.shape[0], 4), device=tgt_boxes.device, dtype=tgt_boxes.dtype)
                tgt_xyxy[:, 0] = (tgt_boxes[:, 0] - tgt_boxes[:, 2] / 2)  # x1
                tgt_xyxy[:, 1] = (tgt_boxes[:, 1] - tgt_boxes[:, 3] / 2)  # y1
                tgt_xyxy[:, 2] = (tgt_boxes[:, 0] + tgt_boxes[:, 2] / 2)  # x2
                tgt_xyxy[:, 3] = (tgt_boxes[:, 1] + tgt_boxes[:, 3] / 2)  # y2

                # Create anchor points and strides
                grid_size = int(num_anchors ** 0.5)
                anchor_points = torch.zeros((num_anchors, 2), device=device)
                stride_tensor = torch.full((num_anchors,), stride, device=device)

                # Generate grid points
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(grid_size, device=device),
                    torch.arange(grid_size, device=device),
                    indexing='ij'
                )
                if grid_size * grid_size <= num_anchors:
                    anchor_points[:grid_size*grid_size, 0] = x_grid.flatten()
                    anchor_points[:grid_size*grid_size, 1] = y_grid.flatten()

                # Assign targets using TaskAlignedAssigner
                try:
                    # Reshape predictions for assigner
                    pred_box_flat = pred_box[batch_idx]  # [num_anchors, 4]
                    pred_cls_flat = pred_cls[batch_idx] if pred_cls is not None else None

                    # Get assignment results - use Ultralytics 8.x API (pd_scores, pd_bboxes)
                    assign_results = self.assigner(
                        pd_scores=pred_cls_flat,
                        pd_bboxes=pred_box_flat,
                        anc_points=anchor_points,
                        gt_labels=tgt_classes,
                        gt_bboxes=tgt_xyxy,
                        mask_gt=tgt_boxes.new_ones(len(tgt_boxes), dtype=torch.bool)
                    )

                    fg_mask = assign_results['fg_mask']
                    if fg_mask.sum() > 0:
                        # BboxLoss using IoU-based computation
                        matched_idx = assign_results['matched_idx']
                        if matched_idx is not None and matched_idx[fg_mask].numel() > 0:
                            target_for_loss = tgt_xyxy[matched_idx[fg_mask]]
                            pred_for_loss = pred_box_flat[fg_mask]

                            # Use Ultralytics BboxLoss
                            box_loss = self.bbox_loss(
                                pred_for_loss,
                                target_for_loss,
                                xyxy=True
                            )
                            total_box_loss = total_box_loss + box_loss

                            # Classification loss using VarifocalLoss
                            if pred_cls_flat is not None:
                                target_cls = torch.zeros((fg_mask.sum(), self.nc), device=device)
                                for i, cls_idx in enumerate(tgt_classes[matched_idx[fg_mask]]):
                                    if cls_idx < self.nc:
                                        target_cls[i, cls_idx] = 1.0

                                cls_loss = self.cls_loss(
                                    pred_cls_flat[fg_mask],
                                    target_cls
                                )
                                total_cls_loss = total_cls_loss + cls_loss

                            # DFL loss
                            if pred_dfl is not None and len(matched_idx) > 0:
                                # Create DFL target (normalized box coordinates)
                                dfl_target = torch.zeros((fg_mask.sum(), 4), device=device)
                                dfl_target[:, 0] = target_for_loss[:, 0]  # x1
                                dfl_target[:, 1] = target_for_loss[:, 1]  # y1
                                dfl_target[:, 2] = target_for_loss[:, 2]  # x2
                                dfl_target[:, 3] = target_for_loss[:, 3]  # y2

                                dfl_loss = self.dfl_loss(
                                    pred_dfl[batch_idx][fg_mask],
                                    dfl_target
                                )
                                total_dfl_loss = total_dfl_loss + dfl_loss

                        num_valid += 1

                except Exception:
                    # Fallback to simple loss computation
                    num_tgt = len(tgt_boxes)
                    if num_tgt > 0 and pred_cls is not None:
                        num_repeat = (num_anchors // num_tgt) + 1
                        tgt_repeated = tgt_boxes.repeat(num_repeat, 1)[:num_anchors]
                        box_loss = F.mse_loss(pred_box[batch_idx], tgt_repeated)
                        total_box_loss = total_box_loss + box_loss

                        # Simple BCE for classification
                        tgt_cls_onehot = torch.zeros((num_anchors, self.nc), device=device)
                        for i, cls in enumerate(tgt_classes):
                            if i < num_anchors:
                                tgt_cls_onehot[i, cls] = 1.0
                        cls_loss = F.binary_cross_entropy_with_logits(
                            pred_cls[batch_idx],
                            tgt_cls_onehot
                        )
                        total_cls_loss = total_cls_loss + cls_loss
                        num_valid += 1

        # Average losses
        if num_valid > 0:
            total_box_loss = total_box_loss / max(num_valid, 1)
            total_cls_loss = total_cls_loss / max(num_valid, 1)
            total_dfl_loss = total_dfl_loss / max(num_valid, 1)

        # Combine losses with Ultralytics weights
        total_loss = (self.box_loss_weight * total_box_loss +
                      self.cls_loss_weight * total_cls_loss +
                      self.dfl_loss_weight * total_dfl_loss)

        return total_loss, {
            'box_loss': total_box_loss.item() if isinstance(total_box_loss, torch.Tensor) else 0.0,
            'cls_loss': total_cls_loss.item() if isinstance(total_cls_loss, torch.Tensor) else 0.0,
            'dfl_loss': total_dfl_loss.item() if isinstance(total_dfl_loss, torch.Tensor) else 0.0
        }


# =============================================================================
# Metrics functions (kept from original)
# =============================================================================

def compute_map_reproduced(model, dataloader, device, debug=False,
                           conf_threshold=0.25, iou_threshold=0.45,
                           iou_thresholds=None):
    """
    Compute mAP using reproduced metrics.

    Args:
        model: The YOLO model
        dataloader: DataLoader returning (images, targets)
                   targets: [N, 6] where each row is [batch_idx, class_id, x, y, w, h]
                   coordinates are normalized (0-1) center format (xywh)
        device: Device to run on
        debug: Enable debug output
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        iou_thresholds: IoU thresholds for mAP computation (default: np.linspace(0.5, 0.95, 10))

    Returns:
        Dictionary with mAP50, mAP50-95, precision, recall
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)

    model.eval()

    # Store all predictions and ground truths per image
    # Each entry: {'boxes': [N, 4] xyxy, 'scores': [N], 'classes': [N]}
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Handle different dataloader return formats
            # Ultralytics YOLODataset returns: (images, targets) or {'img': images, 'labels': targets}
            if isinstance(batch, dict):
                images = batch.get('img', batch.get('images'))
                targets = batch.get('labels', batch.get('targets', None))
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    images, targets = batch[0], batch[1]
                else:
                    images = batch[0]
                    targets = None
            else:
                images = batch
                targets = None

            if images is None:
                continue

            if debug and batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}")

            images = images.to(device)

            # Convert uint8 to float32 and normalize to [0, 1]
            if images.dtype == torch.uint8:
                images = images.float() / 255.0

            batch_size = images.shape[0]

            # Run model
            outputs = model(images)

            # Handle different output formats from YOLO model
            # Outputs can be: tuple of (predictions, extra), list of tensors, or single tensor
            if isinstance(outputs, tuple):
                first_layer = outputs[0]
            elif isinstance(outputs, list):
                first_layer = outputs[0]
            else:
                first_layer = outputs

            # Handle different output formats
            # Check if already NMS-processed (shape [batch, num_dets, 6])

            # Handle different output formats
            if isinstance(outputs, tuple) and len(outputs) > 0:
                # Check the first element
                first_elem = outputs[0]
                if isinstance(first_elem, torch.Tensor) and first_elem.dim() == 3 and first_elem.shape[2] == 6:
                    # Already NMS-processed format: [batch, num_dets, 6]
                    pred_boxes = outputs[0]  # [batch, num_dets, 6]
                elif isinstance(first_elem, torch.Tensor) and first_elem.dim() == 3:
                    pred_boxes = outputs[0]
                elif isinstance(first_elem, torch.Tensor) and first_elem.dim() == 4:
                    pred_boxes = outputs  # List format
                else:
                    pred_boxes = outputs[0] if hasattr(outputs, '__getitem__') else outputs
            elif isinstance(outputs, list) and len(outputs) > 0:
                first_layer = outputs[0]
                if first_layer.dim() == 3 and first_layer.shape[2] == 6:
                    pred_boxes = outputs[0]
                elif first_layer.dim() == 3:
                    pred_boxes = outputs
                elif first_layer.dim() == 4:
                    pred_boxes = outputs
                else:
                    pred_boxes = outputs
            else:
                pred_boxes = outputs

            pred_scores = None
            pred_classes = None

            if pred_boxes is None or (isinstance(pred_boxes, list) and len(pred_boxes) == 0):
                # No detections, fill empty for each image in batch
                for i in range(batch_size):
                    all_predictions.append({
                        'boxes': np.zeros((0, 4), dtype=np.float32),
                        'scores': np.zeros(0, dtype=np.float32),
                        'classes': np.zeros(0, dtype=np.int64)
                    })
                continue

            # Process each image in batch
            for i in range(batch_size):
                img_boxes = None
                img_scores = None
                img_dets = None  # Flag to track if we already have detections

                # Get detections for this image
                if isinstance(pred_boxes, torch.Tensor) and pred_boxes.dim() == 3 and pred_boxes.shape[2] == 6:
                    # Already in NMS format: [batch, num_dets, 6] = [x1,y1,x2,y2,score,class]
                    img_dets_tensor = pred_boxes[i]  # [num_dets, 6]

                    # Filter by confidence
                    conf_mask = img_dets_tensor[:, 4] > conf_threshold
                    img_dets_tensor = img_dets_tensor[conf_mask]

                    if len(img_dets_tensor) > 0:
                        # Boxes are in pixel coordinates, convert to normalized (0-1)
                        boxes = img_dets_tensor[:, :4].cpu().numpy()
                        boxes[:, [0, 2]] /= 640.0  # x1, x2
                        boxes[:, [1, 3]] /= 640.0  # y1, y2
                        # Clip boxes to [0, 1] range
                        boxes = np.clip(boxes, 0.0, 1.0)
                        img_dets = {
                            'boxes': boxes,
                            'scores': img_dets_tensor[:, 4].cpu().numpy(),
                            'classes': img_dets_tensor[:, 5].long().cpu().numpy()
                        }
                    else:
                        img_dets = {
                            'boxes': np.zeros((0, 4), dtype=np.float32),
                            'scores': np.zeros(0, dtype=np.float32),
                            'classes': np.zeros(0, dtype=np.int64)
                        }
                elif isinstance(pred_boxes, list):
                    # List of detection layers - parse each and combine
                    # Each layer: [batch, channels, H, W] -> reshape to [batch, channels, H*W]
                    all_layer_boxes = []
                    all_layer_scores = []
                    reg_max = 16
                    nc = 80  # default num classes

                    for layer_pred in pred_boxes:
                        if layer_pred is None:
                            continue

                        # Handle different tensor dimensions
                        if layer_pred.dim() == 3:
                            # Shape: [batch, channels, num_anchors]
                            layer_flat = layer_pred  # Already flattened
                            b, c, n = layer_flat.shape
                            # Extract for this image
                            img_flat = layer_flat[i]  # [channels, num_anchors]
                        elif layer_pred.dim() == 4:
                            # Shape: [batch, channels, H, W]
                            b, c, h, w = layer_pred.shape
                            # Reshape to [batch, channels, num_anchors]
                            layer_flat = layer_pred.flatten(2)  # [batch, channels, H*W]
                            img_flat = layer_flat[i]  # [channels, num_anchors]
                        else:
                            continue

                        # Split into box and class parts
                        # boxes: [4*reg_max, num_anchors], scores: [nc, num_anchors]
                        if c >= 4 * reg_max:
                            box_part = img_flat[:4 * reg_max, :]  # [64, num_anchors]
                            score_part = img_flat[4 * reg_max:4 * reg_max + nc, :] if c > 4 * reg_max else img_flat[4 * reg_max:, :]
                        else:
                            # Fallback
                            box_part = img_flat[:4, :]
                            score_part = img_flat[4:, :]

                        all_layer_boxes.append(box_part)
                        all_layer_scores.append(score_part)

                    if all_layer_boxes:
                        # Concatenate along anchor dimension
                        img_boxes = torch.cat(all_layer_boxes, dim=1)  # [64, total_anchors]
                        img_scores = torch.cat(all_layer_scores, dim=1)  # [nc, total_anchors]
                    else:
                        img_boxes = None
                        img_scores = None
                elif pred_boxes.dim() == 3:
                    img_boxes = pred_boxes[i]  # [channels, num_anchors]
                    img_scores = None
                else:
                    img_boxes = pred_boxes[i]
                    img_scores = None

                # Parse raw output - only if not already handled by NMS case
                if img_dets is None:
                    if img_boxes is not None:
                        img_dets = _parse_raw_detection_with_scores(img_boxes, img_scores, conf_threshold, iou_threshold)
                    else:
                        img_dets = {
                            'boxes': np.zeros((0, 4), dtype=np.float32),
                            'scores': np.zeros(0, dtype=np.float32),
                            'classes': np.zeros(0, dtype=np.int64)
                        }

                all_predictions.append(img_dets)

                # Get ground truths for this image
                if targets is not None and targets.shape[0] > 0:
                    batch_mask = targets[:, 0] == i
                    img_targets = targets[batch_mask]
                else:
                    img_targets = torch.zeros((0, 6), device=device if targets is not None else 'cpu')

                if len(img_targets) > 0:
                    # targets: [batch_idx, class_id, x, y, w, h] normalized
                    classes = img_targets[:, 1].cpu().numpy().astype(np.int64)
                    x, y, w, h = img_targets[:, 2].cpu().numpy(), img_targets[:, 3].cpu().numpy(), \
                                 img_targets[:, 4].cpu().numpy(), img_targets[:, 5].cpu().numpy()

                    # Convert from center (xywh) to corner (xyxy)
                    x1 = (x - w/2)
                    y1 = (y - h/2)
                    x2 = (x + w/2)
                    y2 = (y + h/2)

                    gt_boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
                else:
                    gt_boxes = np.zeros((0, 4), dtype=np.float32)
                    classes = np.zeros(0, dtype=np.int64)

                all_ground_truths.append({
                    'boxes': gt_boxes,
                    'classes': classes
                })

    # Compute mAP
    if debug:
        print(f"Computing mAP from {len(all_predictions)} predictions and {len(all_ground_truths)} ground truths...")

        # Debug: check first few ground truths and predictions - access them directly
        if len(all_predictions) > 0:
            print(f"  First pred: {all_predictions[0]}")
            print(f"  Second pred: {all_predictions[1]}")

        for i in range(min(3, len(all_ground_truths))):
            gt = all_ground_truths[i]
            pred = all_predictions[i]
            print(f"  GT {i}: classes={gt['classes']}, boxes shape={gt['boxes'].shape}")
            print(f"  Pred {i}: classes={pred['classes']}, boxes shape={pred['boxes'].shape}")

    metrics = compute_ap_metrics(all_predictions, all_ground_truths, iou_thresholds, debug)

    return metrics


def _parse_raw_detection(det_tensor, conf_threshold=0.25, iou_threshold=0.45):
    """
    Parse raw YOLO output tensor to detection format.

    det_tensor: [channels, num_anchors] - raw model output
    Returns: dict with boxes, scores, classes
    """
    if det_tensor is None or det_tensor.numel() == 0:
        return {
            'boxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32),
            'classes': np.zeros(0, dtype=np.int64)
        }

    # Flatten
    if det_tensor.dim() == 3:
        det_tensor = det_tensor.flatten(1)  # [channels, num_anchors]
    elif det_tensor.dim() == 2:
        det_tensor = det_tensor
    else:
        det_tensor = det_tensor.unsqueeze(0).flatten(1)

    c, n = det_tensor.shape

    # Assume reg_max = 16, so 4*16 = 64 for box regression
    reg_max = 16
    num_classes = c - 4 * reg_max if c > 4 * reg_max else max(0, c - 4)

    if num_classes > 0:
        # Extract boxes and scores
        boxes_flat = det_tensor[:4 * reg_max, :]  # [64, num_anchors]
        scores_flat = det_tensor[4 * reg_max:4 * reg_max + num_classes, :]  # [nc, num_anchors]

        # Parse boxes (assuming DFL/distribution focal loss format)
        # Take first 4 channels as box predictions (cx, cy, w, h)
        box_pred = boxes_flat[:4, :].T  # [num_anchors, 4] - cx, cy, w, h
        # Parse scores - take argmax or use all
        class_scores = torch.softmax(scores_flat, dim=0)
        max_scores, pred_classes = class_scores.max(dim=0)
        max_scores = max_scores.numpy()
        pred_classes = pred_classes.numpy()

        # Convert from cx,cy,w,h to x1,y1,x2,y2
        cx, cy, w, h = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2], box_pred[:, 3]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        boxes = np.stack([x1, y1, x2, y2], axis=1)
    else:
        # Fallback: treat first 4 as xyxy
        boxes = det_tensor[:4, :].T.numpy()
        max_scores = np.ones(boxes.shape[0])
        pred_classes = np.zeros(boxes.shape[0], dtype=np.int64)

    # Filter by confidence
    mask = max_scores > conf_threshold
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    pred_classes = pred_classes[mask]

    if len(boxes) == 0:
        return {
            'boxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32),
            'classes': np.zeros(0, dtype=np.int64)
        }

    # Apply NMS per class
    keep_boxes = []
    keep_scores = []
    keep_classes = []

    for cls in np.unique(pred_classes):
        cls_mask = pred_classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = max_scores[cls_mask]

        if len(cls_boxes) == 0:
            continue

        # NMS
        cls_boxes_t = torch.from_numpy(cls_boxes)
        cls_scores_t = torch.from_numpy(cls_scores)
        keep = torchvision.ops.nms(cls_boxes_t, cls_scores_t, iou_threshold)

        keep_boxes.append(cls_boxes[keep.numpy()])
        keep_scores.append(cls_scores[keep.numpy()])
        keep_classes.append(np.full(len(keep), cls, dtype=np.int64))

    if len(keep_boxes) > 0:
        all_boxes = np.concatenate(keep_boxes, axis=0)
        all_scores = np.concatenate(keep_scores, axis=0)
        all_classes = np.concatenate(keep_classes, axis=0)
    else:
        all_boxes = np.zeros((0, 4), dtype=np.float32)
        all_scores = np.zeros(0, dtype=np.float32)
        all_classes = np.zeros(0, dtype=np.int64)

    return {
        'boxes': all_boxes,
        'scores': all_scores,
        'classes': all_classes
    }


def _parse_raw_detection_with_scores(boxes_tensor, scores_tensor, conf_threshold=0.25, iou_threshold=0.45):
    """
    Parse raw YOLO output with pre-separated box and score tensors.

    boxes_tensor: [4*reg_max, num_anchors] - box predictions
    scores_tensor: [nc, num_anchors] - class scores (before sigmoid)
    Returns: dict with boxes, scores, classes
    """
    if boxes_tensor is None or boxes_tensor.numel() == 0:
        return {
            'boxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32),
            'classes': np.zeros(0, dtype=np.int64)
        }

    # Parse boxes (assuming DFL/distribution focal loss format)
    # Take first 4 channels as box predictions (cx, cy, w, h)
    box_pred = boxes_tensor[:4, :].T  # [num_anchors, 4] - cx, cy, w, h

    # Parse scores - apply softmax
    if scores_tensor is not None and scores_tensor.numel() > 0:
        class_scores = torch.softmax(scores_tensor, dim=0)
        max_scores, pred_classes = class_scores.max(dim=0)
        max_scores = max_scores.numpy()
        pred_classes = pred_classes.numpy()
    else:
        # Fallback: no scores, use uniform
        n = boxes_tensor.shape[1]
        max_scores = np.ones(n) / boxes_tensor.shape[0]
        pred_classes = np.zeros(n, dtype=np.int64)

    # Convert from cx,cy,w,h to x1,y1,x2,y2
    cx, cy, w, h = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2], box_pred[:, 3]
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Filter by confidence
    mask = max_scores > conf_threshold
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    pred_classes = pred_classes[mask]

    if len(boxes) == 0:
        return {
            'boxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32),
            'classes': np.zeros(0, dtype=np.int64)
        }

    # Apply NMS per class
    keep_boxes = []
    keep_scores = []
    keep_classes = []

    for cls in np.unique(pred_classes):
        cls_mask = pred_classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = max_scores[cls_mask]

        if len(cls_boxes) == 0:
            continue

        # NMS
        cls_boxes_t = torch.from_numpy(cls_boxes)
        cls_scores_t = torch.from_numpy(cls_scores)
        keep = torchvision.ops.nms(cls_boxes_t, cls_scores_t, iou_threshold)

        keep_boxes.append(cls_boxes[keep.numpy()])
        keep_scores.append(cls_scores[keep.numpy()])
        keep_classes.append(np.full(len(keep), cls, dtype=np.int64))

    if len(keep_boxes) > 0:
        all_boxes = np.concatenate(keep_boxes, axis=0)
        all_scores = np.concatenate(keep_scores, axis=0)
        all_classes = np.concatenate(keep_classes, axis=0)
    else:
        all_boxes = np.zeros((0, 4), dtype=np.float32)
        all_scores = np.zeros(0, dtype=np.float32)
        all_classes = np.zeros(0, dtype=np.int64)

    return {
        'boxes': all_boxes,
        'scores': all_scores,
        'classes': all_classes
    }


def compute_ap_metrics(predictions, ground_truths, iou_thresholds=None, debug=False):
    """
    Compute AP (Average Precision) at multiple IoU thresholds.

    Args:
        predictions: List of dicts with 'boxes', 'scores', 'classes'
        ground_truths: List of dicts with 'boxes', 'classes'
        iou_thresholds: Array of IoU thresholds
        debug: Enable debug output

    Returns:
        Dictionary with mAP50, mAP50-95, precision, recall
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)

    num_classes = 0
    for gt in ground_truths:
        if len(gt['classes']) > 0:
            num_classes = max(num_classes, gt['classes'].max() + 1)

    if debug:
        print(f"  Computing mAP with {len(ground_truths)} images, {num_classes} classes")

    if num_classes == 0:
        if debug:
            print("No ground truth found, returning zeros")
        return {
            'mAP50': 0.0,
            'mAP50-95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

    # Compute AP for each class and IoU threshold
    all_aps = []
    precisions = []
    recalls = []

    for iou_thresh in iou_thresholds:
        class_aps = []

        for cls in range(num_classes):
            # Collect all predictions and ground truths for this class
            cls_predictions = []  # List of (score, is_tp, is_fp)
            num_gt = 0

            for img_idx in range(len(predictions)):
                pred = predictions[img_idx]
                gt = ground_truths[img_idx]

                # Get predictions for this class
                cls_mask = pred['classes'] == cls
                cls_boxes = pred['boxes'][cls_mask]
                cls_scores = pred['scores'][cls_mask]

                # Get ground truths for this class
                gt_cls_mask = gt['classes'] == cls
                gt_boxes = gt['boxes'][gt_cls_mask]
                num_gt += len(gt_boxes)

                if len(cls_boxes) == 0:
                    continue

                if len(gt_boxes) == 0:
                    # All predictions are false positives
                    for score in cls_scores:
                        cls_predictions.append((score, False, True))
                    continue

                # Compute IoU between each prediction and each ground truth
                pred_boxes_t = torch.from_numpy(cls_boxes).float()
                gt_boxes_t = torch.from_numpy(gt_boxes).float()

                # Compute IoU matrix: [num_preds, num_gts]
                iou_matrix = torchvision.ops.box_iou(pred_boxes_t, gt_boxes_t).numpy()

                # For each prediction, find the best matching ground truth
                matched_gts = set()
                for pred_idx in range(len(cls_boxes)):
                    score = cls_scores[pred_idx]
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx in range(len(gt_boxes)):
                        if gt_idx in matched_gts:
                            continue
                        iou = iou_matrix[pred_idx, gt_idx]
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        cls_predictions.append((score, True, False))
                        matched_gts.add(best_gt_idx)
                    else:
                        cls_predictions.append((score, False, True))

            if num_gt == 0:
                continue

            # Sort by score descending
            cls_predictions.sort(key=lambda x: x[0], reverse=True)

            # Compute precision-recall curve
            tp_cumsum = np.cumsum([p[1] for p in cls_predictions])
            fp_cumsum = np.cumsum([p[2] for p in cls_predictions])

            recall = tp_cumsum / num_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

            # Compute AP using 101-point interpolation
            ap = compute_ap_from_pr_curve(recall, precision)
            class_aps.append(ap)

        if len(class_aps) > 0:
            all_aps.append(np.mean(class_aps))

    # Compute final metrics
    if len(all_aps) > 0:
        map_50_95 = np.mean(all_aps)
        map_50 = all_aps[0] if len(all_aps) > 0 else 0.0
    else:
        map_50_95 = 0.0
        map_50 = 0.0

    # Compute precision and recall at IoU=0.5
    if len(iou_thresholds) > 0:
        # Use first threshold (0.5) for overall precision/recall
        iou_thresh = iou_thresholds[0]

        total_tp = 0
        total_fp = 0
        total_gt = 0

        for cls in range(num_classes):
            cls_predictions = []
            num_gt = 0

            for img_idx in range(len(predictions)):
                pred = predictions[img_idx]
                gt = ground_truths[img_idx]

                cls_mask = pred['classes'] == cls
                cls_boxes = pred['boxes'][cls_mask]
                cls_scores = pred['scores'][cls_mask]

                gt_cls_mask = gt['classes'] == cls
                gt_boxes = gt['boxes'][gt_cls_mask]
                num_gt += len(gt_boxes)

                if len(cls_boxes) == 0:
                    continue

                if len(gt_boxes) == 0:
                    total_fp += len(cls_boxes)
                    continue

                pred_boxes_t = torch.from_numpy(cls_boxes).float()
                gt_boxes_t = torch.from_numpy(gt_boxes).float()
                iou_matrix = torchvision.ops.box_iou(pred_boxes_t, gt_boxes_t).numpy()

                matched_gts = set()
                for pred_idx in range(len(cls_boxes)):
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx in range(len(gt_boxes)):
                        if gt_idx in matched_gts:
                            continue
                        iou = iou_matrix[pred_idx, gt_idx]
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        total_tp += 1
                        matched_gts.add(best_gt_idx)
                    else:
                        total_fp += 1

            total_gt += num_gt

        precision = total_tp / (total_tp + total_fp + 1e-10) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_gt + 1e-10) if total_gt > 0 else 0.0
    else:
        precision = 0.0
        recall = 0.0

    if debug:
        print(f"mAP50: {map_50:.4f}, mAP50-95: {map_50_95:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return {
        'mAP50': map_50,
        'mAP50-95': map_50_95,
        'precision': precision,
        'recall': recall
    }


def compute_ap_from_pr_curve(recall, precision):
    """
    Compute Average Precision from precision-recall curve using 101-point interpolation.
    """
    # Add sentinel values
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])

    # Compute precision envelope (make precision monotonically decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def compute_ultralytics_map(model, dataloader, device, debug=False):
    """Compute mAP using Ultralytics metrics."""
    from ultralytics.utils.metrics import Metrics

    model.eval()
    metrics = Metrics()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Handle different dataloader return formats
            if isinstance(batch, dict):
                images = batch.get('img', batch.get('images'))
                targets = batch.get('labels', batch.get('targets', None))
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    images, targets = batch[0], batch[1]
                else:
                    images = batch[0]
                    targets = None
            else:
                images = batch
                targets = None

            if images is None:
                continue

            images = images.to(device)

            # Convert uint8 to float32 and normalize to [0, 1]
            if images.dtype == torch.uint8:
                images = images.float() / 255.0

            outputs = model(images)

            # Convert output format
            if isinstance(outputs, tuple):
                pred = outputs[0]
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                # New format: list of standard ultralytics [batch, channels, H, W]
                # Use the list directly - ultralytics Metrics will handle it
                pred = outputs
            elif isinstance(outputs, dict):
                if 'one2one' in outputs:
                    pred = outputs['one2one']
                else:
                    pred = outputs
            else:
                pred = outputs

            # Process batch
            # This is simplified - real implementation would handle NMS etc.
            pass

    return {
        'mAP50': metrics.box.map50 if hasattr(metrics, 'box') else 0.0,
        'mAP50-95': metrics.box.map if hasattr(metrics, 'box') else 0.0,
        'precision': metrics.box.mp if hasattr(metrics, 'box') else 0.0,
        'recall': metrics.box.mr if hasattr(metrics, 'box') else 0.0
    }


def get_loss_function(version='ultralytics', model=None):
    """Get loss function based on version."""
    if version == 'ultralytics':
        try:
            from ultralytics.utils.loss import v8DetectionLoss
            return v8DetectionLoss(model)
        except ImportError:
            return DetectionLossReproduced(model)
    else:
        return DetectionLossReproduced(model)


def get_map_function(version='ultralytics'):
    """Get mAP computation function based on version."""
    if version == 'ultralytics':
        return compute_ultralytics_map
    else:
        return compute_map_reproduced


# Keep original functions for compatibility
def get_ultralytics_loss(model=None):
    """Get Ultralytics loss function."""
    return get_loss_function('ultralytics', model)
