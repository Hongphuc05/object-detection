import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.config import DEFAULT_CLASSES, detect_device, extract_model_state, torch_load_compat
from utils.detector import Detector, decode_predictions
from utils.nms import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser(description="Run object detection inference and output predictions.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to evaluate.")
    parser.add_argument("--output", type=str, required=True, help="Path to write the predictions JSON file.")
    parser.add_argument("--weights", type=str, default="./models/best.pth", help="Path to model weights checkpoint.")
    parser.add_argument("--img_size", type=int, default=512, help="Input size the model was trained on.")
    parser.add_argument("--conf_thres", type=float, default=0.05, help="Confidence score threshold.")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IoU threshold for NMS.")
    return parser.parse_args()


def preprocess_image(img_path, target_size):
    img = cv2.imread(img_path)
    if img is None:
        return None, 0, 0

    h_orig, w_orig = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std

    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor, h_orig, w_orig


def main():
    args = parse_args()
    device = detect_device()
    print(f"Using device for inference: {device}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model checkpoint weight file not found at: {args.weights}")

    checkpoint = torch_load_compat(args.weights, map_location=device)
    checkpoint_classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
    classes = checkpoint_classes or DEFAULT_CLASSES

    print(f"Loading model with weights from {args.weights}...")
    model = Detector(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(extract_model_state(checkpoint))
    model.eval()

    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, pattern)))

    image_paths = sorted(set(image_paths))
    print(f"Found {len(image_paths)} images to process in {args.image_dir}")

    results = []
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Inference"):
            img_id = os.path.basename(img_path)
            img_tensor, h_orig, w_orig = preprocess_image(img_path, args.img_size)
            if img_tensor is None:
                print(f"Warning: Failed to load image {img_path}, skipping.")
                results.append({"image_id": img_id, "boxes": []})
                continue

            img_tensor = img_tensor.to(device)
            predictions = model(img_tensor)

            strides = {"p3": 8, "p4": 16, "p5": 32}
            all_bboxes = []
            all_obj_scores = []
            all_cls_probs = []
            for level, stride in strides.items():
                cls_pred, reg_pred, obj_pred = predictions[level]
                bboxes, obj_scores, cls_probs = decode_predictions(reg_pred, obj_pred, cls_pred, stride, device)
                all_bboxes.append(bboxes)
                all_obj_scores.append(obj_scores)
                all_cls_probs.append(cls_probs)

            detections = non_max_suppression(
                torch.cat(all_bboxes, dim=1),
                torch.cat(all_obj_scores, dim=1),
                torch.cat(all_cls_probs, dim=1),
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
            )

            boxes_out = []
            for det in detections[0]:
                xmin = float(det[0]) * (w_orig / args.img_size)
                ymin = float(det[1]) * (h_orig / args.img_size)
                xmax = float(det[2]) * (w_orig / args.img_size)
                ymax = float(det[3]) * (h_orig / args.img_size)
                conf = float(det[4])
                cls_id = int(det[5])

                xmin = max(0.0, min(xmin, w_orig))
                ymin = max(0.0, min(ymin, h_orig))
                xmax = max(0.0, min(xmax, w_orig))
                ymax = max(0.0, min(ymax, h_orig))

                if xmax > xmin and ymax > ymin:
                    boxes_out.append(
                        {
                            "class": classes[cls_id],
                            "confidence": round(conf, 4),
                            "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                        }
                    )

            results.append({"image_id": img_id, "boxes": boxes_out})

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print(f"Predictions successfully written to {args.output}")


if __name__ == "__main__":
    main()
