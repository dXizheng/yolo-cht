import torch
import numpy as np


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    boxes1: [N, 4] (x1, y1, x2, y2)
    boxes2: [M, 4] (x1, y1, x2, y2)
    Returns: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter + 1e-6

    return inter / union


def compute_map_metric(detections, targets, iou_threshold=0.5, img_size=640):
    """
    Compute mAP at a specific IoU threshold.

    Args:
        detections: List of [N, 6] (x1, y1, x2, y2, conf, class)
        targets: List of [M, 6] (batch_idx, class, x, y, w, h) - normalized, FLAT per batch
        iou_threshold: IoU threshold
        img_size: Image size for denormalization

    Returns:
        precision, recall, ap
    """
    # Process targets - targets is a flat tensor with all targets from all images
    # targets format: [batch_idx, class, x, y, w, h] normalized
    processed_targets = []
    num_gt = 0

    for i in range(len(detections)):  # Process per image
        # targets is a list of tensors per image
        img_targets = targets[i]

        if img_targets is not None and img_targets.shape[0] > 0:
            # Denormalize: [batch_idx, class, x, y, w, h] -> [class, x1, y1, x2, y2]
            cls = img_targets[:, 1].long()
            x = img_targets[:, 2] * img_size
            y = img_targets[:, 3] * img_size
            w = img_targets[:, 4] * img_size
            h = img_targets[:, 5] * img_size

            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            processed_targets.append((cls, boxes))
            num_gt += img_targets.shape[0]
        else:
            processed_targets.append((None, None))

    if num_gt == 0:
        return 0.0, 0.0, 0.0

    # Collect all predictions sorted by confidence
    all_preds = []
    for img_idx, det in enumerate(detections):
        if det.shape[0] > 0:
            for det_idx in range(det.shape[0]):
                all_preds.append({
                    'img_idx': img_idx,
                    'box': det[det_idx, :4],
                    'conf': det[det_idx, 4].item(),
                    'cls': det[det_idx, 5].long().item()
                })

    # Sort by confidence
    all_preds = sorted(all_preds, key=lambda x: x['conf'], reverse=True)

    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0

    # Track which GT has been matched
    matched_gt = [set() for _ in range(len(targets))]  # img_idx -> set of matched gt indices

    # Count TP and FP
    tp = []
    fp = []

    for pred in all_preds:
        img_idx = pred['img_idx']
        pred_box = pred['box'].unsqueeze(0)  # [1, 4]
        pred_cls = pred['cls']

        # Get targets for this image
        tgt_cls, tgt_boxes = processed_targets[img_idx]

        if tgt_boxes is None or tgt_boxes.shape[0] == 0:
            # No targets for this image
            fp.append(1)
            tp.append(0)
            continue

        # Compute IoU with all targets
        iou = box_iou(pred_box, tgt_boxes)[0]  # [M]

        # Find max IoU with correct class
        cls_mask = (tgt_cls == pred_cls)
        if cls_mask.sum() == 0:
            # No target with matching class
            fp.append(1)
            tp.append(0)
            continue

        iou_cls = iou * cls_mask.float()
        max_iou, max_idx = iou_cls.max(0)
        max_iou = max_iou.item()
        max_idx = max_idx.item()

        # Check if already matched
        if max_idx in matched_gt[img_idx]:
            # Already matched, this is a false positive
            fp.append(1)
            tp.append(0)
        elif max_iou >= iou_threshold:
            # True positive
            matched_gt[img_idx].add(max_idx)
            fp.append(0)
            tp.append(1)
        else:
            # IoU too low
            fp.append(1)
            tp.append(0)

    # Convert to arrays
    tp = np.array(tp, dtype=np.float32)
    fp = np.array(fp, dtype=np.float32)

    # Cumulative
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    # Precision and recall
    recall = tp_cum / num_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)

    # Compute AP
    ap = compute_ap_simple(recall, precision)

    return precision[-1] if len(precision) > 0 else 0.0, recall[-1] if len(recall) > 0 else 0.0, ap


def compute_ap_simple(recall, precision):
    """Compute Average Precision using 11-point interpolation."""
    # Add sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def compute_map_simple(model, dataloader, device, conf_thres=0.25, iou_thres=0.45, max_batches=10):
    """
    Compute mAP on validation data.

    Args:
        model: YOLO model
        dataloader: validation dataloader
        device: device
        conf_thres: confidence threshold for NMS
        iou_thres: IoU threshold for NMS
        max_batches: maximum batches to process

    Returns:
        Dictionary with mAP metrics
    """
    # Import NMS
    import torchvision

    def nms(boxes, scores, iou_thres):
        """NMS on CPU."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order[0].item())
                break

            i = order[0]
            keep.append(i.item())

            xx1 = x1[order[1:]].clamp(min=x1[i].item())
            yy1 = y1[order[1:]].clamp(min=y1[i].item())
            xx2 = x2[order[1:]].clamp(max=x2[i].item())
            yy2 = y2[order[1:]].clamp(max=y2[i].item())

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = (iou <= iou_thres).nonzero(as_tuple=False).squeeze()
            if inds.numel() == 0:
                break
            order = order[inds + 1]

        return torch.tensor(keep, dtype=torch.long)

    model.eval()

    all_detections = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            bs = images.shape[0]
            images = images.to(device)

            # Process targets - split by batch index
            # targets is [N, 6] = [batch_idx, class, x, y, w, h]
            batch_targets = []
            for i in range(bs):
                # Keep all 6 columns including batch_idx for compute_map_metric
                img_tgt = targets[targets[:, 0] == i]  # [class, x, y, w, h] with batch_idx in original position
                batch_targets.append(img_tgt)

            # Run model
            outputs = model(images)

            # Handle output format
            # Output is [B, N, 6] = (x1, y1, x2, y2, conf, class) after NMS - ALREADY XYXY!
            pred_dict = None

            if isinstance(outputs, tuple) and not isinstance(outputs, list):
                pred = outputs[0]  # [B, N, 6]

                # Process each image in batch
                for i in range(bs):
                    # Get predictions for this image
                    img_pred = pred[i]  # [N, 6]

                    # Filter by confidence
                    conf = img_pred[:, 4]
                    mask = conf > conf_thres

                    if mask.sum() == 0:
                        all_detections.append(torch.zeros(0, 6))
                        # Use batch_targets[i] for consistency
                        all_targets.append(batch_targets[i].cpu() if batch_targets[i].shape[0] > 0 else torch.zeros(0, 6))
                        continue

                    img_pred_filtered = img_pred[mask]  # [M, 6]

                    # Already in xyxy format - use as is
                    all_detections.append(img_pred_filtered.cpu())

                    # Add targets - use batch_targets
                    all_targets.append(batch_targets[i])
                continue  # Skip dict processing for tuple format

            # Handle new standard ultralytics format: list of tensors [batch, channels, H, W]
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                # Convert list of detection layers to predictions
                all_boxes = []
                all_scores = []

                for pred in outputs:
                    if pred is None or pred.dim() != 4:
                        continue

                    b, c, h, w = pred.shape
                    # Reshape to [batch, channels, num_anchors]
                    pred_flat = pred.flatten(2)  # [batch, channels, H*W]

                    # Extract box predictions and class scores
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
                    boxes = torch.cat(all_boxes, dim=2)  # [batch, channels, total_anchors]
                    scores = torch.cat(all_scores, dim=2)  # [batch, nc, total_anchors]
                    pred_dict = {'boxes': boxes, 'scores': scores}
                else:
                    pred_dict = {}
            elif 'one2one' in outputs:
                pred_dict = outputs['one2one']
            elif 'one2many' in outputs:
                pred_dict = outputs['one2many']
            else:
                pred_dict = outputs

            # Process pred_dict if it's a dict with boxes and scores
            if isinstance(pred_dict, dict):
                boxes = pred_dict.get('boxes', None)
                scores = pred_dict.get('scores', None)

                if boxes is not None and scores is not None:
                    # [B, 4, N], [B, 80, N]
                    bs = boxes.shape[0]
                    num_classes = scores.shape[1]

                    # Get max scores across classes
                    max_scores, pred_cls = scores.max(dim=1)  # [B, N]

                    # Convert boxes from xywh to xyxy
                    boxes_xyxy = boxes.clone()
                    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
                    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
                    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
                    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

                    # Process each image in batch
                    for i in range(bs):
                        # Filter by confidence
                        mask = max_scores[i] > conf_thres
                        if mask.sum() == 0:
                            all_detections.append(torch.zeros(0, 6))
                            all_targets.append(batch_targets[i])
                            continue

                        img_boxes = boxes_xyxy[i, :, mask].T  # [N, 4]
                        img_scores = max_scores[i, mask]  # [N]
                        img_cls = pred_cls[i, mask]  # [N]

                        # NMS
                        keep = nms(img_boxes.cpu(), img_scores.cpu(), iou_thres)

                        img_boxes = img_boxes[keep]
                        img_scores = img_scores[keep]
                        img_cls = img_cls[keep]

                        # Stack: [N, 6] = x1, y1, x2, y2, conf, class
                        det = torch.stack([
                            img_boxes[:, 0],
                            img_boxes[:, 1],
                            img_boxes[:, 2],
                            img_boxes[:, 3],
                            img_scores,
                            img_cls.float()
                        ], dim=1)

                        all_detections.append(det.cpu())

                        # Add targets - use batch_targets
                        all_targets.append(batch_targets[i].cpu() if batch_targets[i].shape[0] > 0 else torch.zeros(0, 6))
                else:
                    # Fallback
                    for i in range(bs):
                        all_detections.append(torch.zeros(0, 6))
                        all_targets.append(batch_targets[i].cpu() if batch_targets[i].shape[0] > 0 else torch.zeros(0, 6))
            else:
                # Not a dict we understand
                for i in range(bs):
                    all_detections.append(torch.zeros(0, 6))
                    all_targets.append(batch_targets[i].cpu() if batch_targets[i].shape[0] > 0 else torch.zeros(0, 6))

    # Compute mAP at different IoU thresholds
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = []

    for iou_thresh in iou_thresholds:
        _, _, ap = compute_map_metric(all_detections, all_targets, iou_thresh, img_size=640)
        aps.append(ap)

    map50 = aps[0]  # mAP@0.5
    map_avg = np.mean(aps)  # mAP@0.5:0.95

    return {
        'mAP50': map50,
        'mAP50-95': map_avg
    }
