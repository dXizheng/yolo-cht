"""
YOLO26n Inference Script

Inference script for YOLO26n with CHT layers and QAT quantization.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo26n.yolo26n_cht_qat_model import load_yolo26n_cht_qat_model, count_model_params
from yolo26n_config import ReplaceMode


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with YOLO26n CHT + QAT')
    parser.add_argument('--model', type=str, default='yolo26n.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--model-type', type=str, default='cht',
                       choices=['cht', 'standard'],
                       help='Model type: cht (CHT+QAT model) or standard (ultralytics YOLO)')
    parser.add_argument('--source', type=str, default='0',
                       help='Source: image path, video path, or 0 for webcam')
    parser.add_argument('--sparsity', type=float, default=0.9,
                       help='Target sparsity (0.9 = 90%)')
    parser.add_argument('--replace-mode', type=str, default='backbone_neck',
                       choices=['backbone', 'backbone_neck', 'all'],
                       help='Which layers to replace with CHT')
    parser.add_argument('--quantization', type=str, default='int8',
                       choices=['int8', 'fp8', 'none'],
                       help='Quantization type')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='Confidence threshold (default 0.1 for better recall)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--save', action='store_true',
                       help='Save results')
    parser.add_argument('--show', action='store_true',
                       help='Show results')
    return parser.parse_args()


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image while maintaining aspect ratio.

    Args:
        img: Input image
        new_shape: Target shape (width, height)
        color: Padding color

    Returns:
        Resized image and scale factors
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def preprocess_image(img, imgsz=640):
    """
    Preprocess image for inference.

    Args:
        img: Input image (numpy array)
        imgsz: Target image size

    Returns:
        Preprocessed tensor and scale info
    """
    # Letterbox
    img, ratio, pad = letterbox(img, imgsz)

    # Convert to RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, 0)

    # Convert to tensor
    img = torch.from_numpy(img)

    return img, ratio, pad


def postprocess_predictions(preds, conf_thres=0.25, iou_thres=0.45):
    """
    Post-process predictions.

    Args:
        preds: Model predictions (tuple, list, or dict in ultralytics format)
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS

    Returns:
        Detected boxes, scores, class indices
    """
    # Handle tuple output (standard ultralytics inference format: (predictions, info_dict))
    # Predictions are already in [batch, N, 6] = (x1, y1, x2, y2, conf, class)
    if isinstance(preds, tuple):
        preds = preds[0]  # Get the predictions tensor

    # Handle tensor in [batch, N, 6] format (already processed predictions)
    # This is the standard ultralytics inference output
    if isinstance(preds, torch.Tensor) and preds.dim() == 3 and preds.shape[2] == 6:
        # Already in [batch, N, 6] format - use directly
        # Take first batch
        preds = preds[0]  # [N, 6]

    # Handle dict output (legacy format from internal model)
    elif isinstance(preds, dict):
        if 'one2one' in preds:
            pred_dict = preds['one2one']
        elif 'one2many' in preds:
            pred_dict = preds['one2many']
        else:
            pred_dict = preds

        if isinstance(pred_dict, dict):
            feats = pred_dict.get('feats', None)
            if feats is not None and isinstance(feats, list):
                preds = feats
            else:
                boxes = pred_dict.get('boxes', None)
                scores = pred_dict.get('scores', None)
                if boxes is not None and scores is not None:
                    # Convert boxes/scores to list format
                    b = boxes.shape[0] if boxes.dim() >= 3 else 1
                    num_anchors = boxes.shape[2] if boxes.dim() >= 3 else 1
                    h = w = int(num_anchors ** 0.5)
                    if h * w != num_anchors:
                        h, w = 1, num_anchors
                    boxes_reshaped = boxes.view(boxes.shape[0], boxes.shape[1], h, w)
                    preds = [boxes_reshaped]
                else:
                    return [], [], []
        else:
            preds = [pred_dict]

    # Standard ultralytics format: list of [batch, channels, H, W] tensors
    if isinstance(preds, (list, tuple)):
        # Process each detection layer and concatenate
        all_preds = []
        for pred in preds:
            if pred is None or pred.dim() != 4:
                continue

            b, c, h, w = pred.shape
            # Reshape to [batch, channels, num_anchors]
            pred_flat = pred.flatten(2).permute(0, 2, 1)  # [batch, num_anchors, channels]

            # Extract box predictions (first 4*reg_max for DFL) and class scores
            reg_max = 16  # Default reg_max
            # Debug: print channel info to verify reg_max
            # print(f"DEBUG: Total channels={c}, reg_max={reg_max}, num_classes={c - 4 * reg_max}")
            nc = c - 4 * reg_max  # Number of classes

            if nc > 0:
                # Has DFL: [num_anchors, 4*reg_max + nc]
                boxes_flat = pred_flat[:, :, :4 * reg_max]  # [batch, num_anchors, 4*reg_max]
                scores_flat = pred_flat[:, :, 4 * reg_max:]  # [batch, num_anchors, nc]
            else:
                # No DFL: [num_anchors, 4 + nc]
                boxes_flat = pred_flat[:, :, :4]  # [batch, num_anchors, 4]
                scores_flat = pred_flat[:, :, 4:] if pred_flat.shape[2] > 4 else torch.zeros((b, pred_flat.shape[1], 1), device=pred.device)

            # Take first batch
            boxes_flat = boxes_flat[0]  # [num_anchors, 4*reg_max]
            scores_flat = scores_flat[0]  # [num_anchors, nc]

            # Apply DFL (Distribution Focal Loss) to get xyxy boxes
            if boxes_flat.shape[1] > 4:
                # DFL distribution -> convert to xyxy using expected value
                boxes_dfl = boxes_flat.view(-1, 4, reg_max)  # [num_anchors, 4, reg_max]
                # Apply softmax to get distribution over reg_max
                prob = torch.softmax(boxes_dfl, dim=2)  # [num_anchors, 4, reg_max]
                # Compute expected value: sum(prob * position)
                reg_range = torch.arange(reg_max, device=boxes_flat.device, dtype=boxes_flat.dtype)
                boxes_flat = (prob * reg_range).sum(dim=2)  # [num_anchors, 4]

            # Get confidence as max class score (apply sigmoid first)
            scores_flat = torch.sigmoid(scores_flat)
            conf, classes = scores_flat.max(dim=1) if scores_flat.dim() > 1 else (scores_flat, torch.zeros(boxes_flat.shape[0], dtype=torch.long, device=boxes_flat.device))

            # Concatenate boxes + conf + classes
            if scores_flat.dim() > 1:
                det = torch.cat([boxes_flat, conf.unsqueeze(1), classes.unsqueeze(1).float()], dim=1)
            else:
                det = torch.cat([boxes_flat, conf.unsqueeze(1), classes.unsqueeze(1).float()], dim=1)

            all_preds.append(det)

        if not all_preds:
            return [], [], []

        # Concatenate all detection layers
        preds = torch.cat(all_preds, dim=0)
    else:
        # Single tensor - reshape
        if preds.dim() == 4:
            b, c, h, w = preds.shape
            preds = preds.flatten(2).permute(0, 2, 1)[0]  # [num_anchors, channels]

    if preds is None or preds.shape[0] == 0:
        return [], [], []

    # Filter by confidence
    conf_mask = preds[:, 4] > conf_thres
    preds = preds[conf_mask]

    if preds.shape[0] == 0:
        return [], [], []

    # Get boxes, scores, classes
    boxes = preds[:, :4]  # xyxy
    scores = preds[:, 4]
    classes = preds[:, 5]

    # Apply NMS
    boxes_xyxy = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    classes_np = classes.cpu().numpy()

    # Use torchvision nms
    boxes_t = torch.from_numpy(boxes_xyxy).contiguous()
    scores_t = torch.from_numpy(scores_np).contiguous()

    try:
        keep = torchvision.ops.nms(boxes_t, scores_t, iou_thres)
        boxes = boxes_xyxy[keep]
        scores = scores_np[keep]
        classes = classes_np[keep]
    except Exception as e:
        # Fallback: no NMS - convert to numpy arrays
        boxes = boxes_xyxy
        scores = scores_np
        classes = classes_np

    return boxes, scores, classes


# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def run_inference(
    model,
    image_path,
    device='cuda',
    imgsz=640,
    conf_thres=0.1,
    iou_thres=0.45,
    show=False,
    save=False,
    save_dir='runs/inference',
    use_standard=False
):
    """
    Run inference on a single image.

    Args:
        model: YOLO26n model
        image_path: Path to image
        device: Device
        imgsz: Image size
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        show: Show results
        save: Save results
        save_dir: Save directory

    Returns:
        Detections
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Preprocess
    img_tensor, ratio, pad = preprocess_image(img, imgsz)
    img_tensor = img_tensor.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        preds = model(img_tensor)

    # Post-process
    boxes, scores, classes = postprocess_predictions(preds, conf_thres, iou_thres)

    # Convert boxes to numpy array if needed
    if len(boxes) > 0:
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
        # Ensure boxes is 2D array
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if boxes.dtype != np.float32 and boxes.dtype != np.float64:
            boxes = boxes.astype(np.float32)

        # Also ensure scores and classes are numpy arrays
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        if not isinstance(classes, np.ndarray):
            classes = np.array(classes)
        if classes.ndim == 0:
            classes = classes.reshape(1)

        # Scale boxes back to original image size
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= ratio

    # Draw results
    if show or save:
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            cls_int = int(cls)
            cls_name = COCO_CLASSES[cls_int] if cls_int < len(COCO_CLASSES) else f"class_{cls_int}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name}: {score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if save:
        model_prefix = 'cht' if not use_standard else 'standard'
        save_path = Path(save_dir) / f"{model_prefix}_{Path(image_path).name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)
        print(f"Saved to {save_path}")

    if show:
        cv2.imshow('YOLO26n', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print detection results to console
    print("\nDetection results:")
    if len(boxes) > 0:
        for box, score, cls in zip(boxes, scores, classes):
            cls_int = int(cls)
            cls_name = COCO_CLASSES[cls_int] if cls_int < len(COCO_CLASSES) else f"class_{cls_int}"
            x1, y1, x2, y2 = box
            print(f"  {cls_name}: {score:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    else:
        print("  No detections")

    return {
        'boxes': boxes,
        'scores': scores,
        'classes': classes
    }


def run_video_inference(
    model,
    video_path,
    device='cuda',
    imgsz=640,
    conf_thres=0.1,
    iou_thres=0.45,
    show=False,
    save=False,
    save_dir='runs/inference',
    use_standard=False
):
    """
    Run inference on video.

    Args:
        model: YOLO26n model
        video_path: Path to video
        device: Device
        imgsz: Image size
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        show: Show results
        save: Save results
        save_dir: Save directory
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path) if video_path != '0' else 0)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer
    writer = None
    if save:
        save_path = Path(save_dir) / f"{Path(video_path).stem}_result.mp4"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

    # Process frames
    frame_count = 0
    model.eval()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}...")

            # Preprocess
            img_tensor, ratio, pad = preprocess_image(frame, imgsz)
            img_tensor = img_tensor.to(device)

            # Inference
            preds = model(img_tensor)

            # Post-process
            boxes, scores, classes = postprocess_predictions(preds, conf_thres, iou_thres)

            # Scale boxes
            if len(boxes) > 0:
                boxes[:, [0, 2]] -= pad[0]
                boxes[:, [1, 3]] -= pad[1]
                boxes /= ratio

            # Draw results
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class {int(cls)}: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show/save
            if show:
                cv2.imshow('YOLO26n', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            if writer:
                writer.write(frame)

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")


def main():
    """Main inference function."""
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model based on type
    print("\nLoading model...")
    use_standard = args.model_type == 'standard'

    if use_standard:
        # Load standard Ultralytics YOLO model
        from ultralytics import YOLO
        print(f"Loading standard YOLO model: {args.model}")
        yolo_wrapper = YOLO(args.model)
        model = yolo_wrapper.model.to(device)
        model.eval()

        print(f"Model type: Standard YOLO")
        print(f"Model loaded successfully")
    else:
        # Load CHT model
        from ultralytics import YOLO
        print(f"Loading CHT YOLO model: {args.model}")

        # Check if it's a checkpoint file (from train.py)
        is_checkpoint = False
        checkpoint_sd = None
        try:
            ckpt = torch.load(args.model, map_location='cpu')
            if isinstance(ckpt, dict):
                # Check for either prefix - checkpoint has _model. prefix for CHT layers
                if 'model_state_dict' in ckpt:
                    sd = ckpt['model_state_dict']
                    has_model_keys = any(k.startswith('_model.model.') or k.startswith('original_model.model.') for k in sd.keys())
                    if has_model_keys:
                        is_checkpoint = True
                        checkpoint_sd = sd
                elif any(k.startswith('_model.model.') or k.startswith('original_model.model.') for k in ckpt.keys()):
                    is_checkpoint = True
                    checkpoint_sd = ckpt
        except:
            pass

        if is_checkpoint:
            # Load from checkpoint
            print("Loading from checkpoint...")

            # Use direct key mapping - checkpoint keys match model keys exactly
            # This works because both use _model. and original_model. prefixes
            state_dict = checkpoint_sd

            # Use configuration that matches checkpoint (112 CHT layers)
            replace_inside_attention = True

            model = load_yolo26n_cht_qat_model(
                model_path="yolo26n.pt",  # Use base model for architecture
                sparsity=args.sparsity,
                replace_mode=ReplaceMode.BACKBONE_NECK,
                quantization=args.quantization,
                regrow_method="L3n",
                shared_mask_sw=True,
                soft=True,
                link_update_ratio=0.1,
                skip_first_n_convs=2,
                replace_inside_attention=replace_inside_attention,
                sparsity_schedule='step'
            )

            # Load state dict
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint weights")
            print(f"  Missing keys: {len(missing)}")
            print(f"  Unexpected keys: {len(unexpected)}")
        else:
            # Convert replace mode
            if args.replace_mode == 'backbone':
                replace_mode = ReplaceMode.BACKBONE
            elif args.replace_mode == 'backbone_neck':
                replace_mode = ReplaceMode.BACKBONE_NECK
            else:
                replace_mode = ReplaceMode.ALL

            model = load_yolo26n_cht_qat_model(
                model_path=args.model,
                sparsity=args.sparsity,
                replace_mode=replace_mode,
                quantization=args.quantization,
                regrow_method="L3n",
                shared_mask_sw=True,
                soft=True,
                link_update_ratio=0.1
            )

        # Count parameters
        total_params, trainable_params = count_model_params(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"CHT layers: {model.get_num_cht_layers()}")
        print(f"Target sparsity: {model.get_sparsity_target() * 100}%")

        model = model.to(device)
        model.eval()

        # Enable quantization simulation for true INT8 inference
        if args.quantization != 'none':
            model.set_simulate_quant(True)
            print(f"Enabled quantization simulation for INT8 inference")

    # Print configuration
    print("\n" + "=" * 60)
    print(f"{'CHT' if not use_standard else 'Standard'} YOLO Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model type: {args.model_type}")
    print(f"Source: {args.source}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    if not use_standard:
        print(f"Sparsity: {args.sparsity * 100}%")
        print(f"Replace mode: {args.replace_mode}")
        print(f"Quantization: {args.quantization}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print("=" * 60)

    # Determine source type
    source = args.source

    if Path(source).is_file():
        # Image or video file
        ext = Path(source).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"\nRunning inference on image: {source}")
            run_inference(
                model, source, device, args.imgsz, args.conf, args.iou,
                args.show, args.save, use_standard=use_standard
            )
        elif ext in ['.mp4', '.avi', '.mov']:
            print(f"\nRunning inference on video: {source}")
            run_video_inference(
                model, source, device, args.imgsz, args.conf, args.iou,
                args.show, args.save, use_standard=use_standard
            )
        else:
            print(f"Unsupported file format: {ext}")
    elif source == '0' or source.isdigit():
        # Webcam
        print(f"\nRunning inference on webcam: {source}")
        run_video_inference(
            model, source, device, args.imgsz, args.conf, args.iou,
            args.show, args.save, use_standard=use_standard
        )
    else:
        print(f"Invalid source: {source}")

    print("\nInference completed!")


if __name__ == "__main__":
    # Import torchvision for NMS
    import torchvision
    main()
