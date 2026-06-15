import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import (
    CHECKPOINT_FILES,
    configure_torch_runtime,
    detect_device,
    extract_model_state,
    get_dataloader_kwargs,
    get_gpu_name,
    resolve_batch_size,
    resolve_checkpoint_path,
    resolve_num_workers,
    torch_load_compat,
)
from utils.dataset import ObjectDetectionDataset, collate_fn
from utils.detector import Detector, decode_predictions
from utils.loss import ComputeLoss
from utils.nms import non_max_suppression


def get_autocast_context(device, enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def get_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom object detection model.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train annotations json.")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val annotations json.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to train images directory.")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Path to val images directory.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save model checkpoints.")

    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size. Auto-tuned when omitted.")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers. Auto-tuned when omitted.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience on val loss.")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to a checkpoint file or directory. If omitted, train.py auto-loads an existing checkpoint in checkpoint_dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer, scheduler, scaler, and epoch state from a training checkpoint.",
    )
    parser.add_argument(
        "--map_interval",
        type=int,
        default=5,
        help="Run mAP evaluation every N epochs. Set to 0 to disable periodic mAP evaluation.",
    )
    parser.add_argument("--mosaic_prob", type=float, default=0.25, help="Probability of mosaic augmentation.")
    parser.add_argument("--mixup_prob", type=float, default=0.10, help="Probability of mixup augmentation.")
    parser.add_argument(
        "--close_mosaic_epochs",
        type=int,
        default=5,
        help="Disable mosaic/mixup for the last N epochs so the model fine-tunes on normal images.",
    )

    return parser.parse_args()


def build_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, best_map, classes, args):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "best_map": best_map,
        "classes": classes,
        "args": vars(args),
    }


def save_checkpoint(path, checkpoint):
    torch.save(checkpoint, path)


def maybe_load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device, resume):
    checkpoint = torch_load_compat(checkpoint_path, map_location=device)
    model.load_state_dict(extract_model_state(checkpoint))

    start_epoch = 0
    best_val_loss = float("inf")
    best_map = 0.0

    if resume and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler is not None and scaler_state:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_map = float(checkpoint.get("best_map", best_map))

    return start_epoch, best_val_loss, best_map


def train_one_epoch(model, dataloader, optimizer, compute_loss, scaler, device, use_amp, use_channels_last):
    model.train()
    running_loss = 0.0
    running_reg = 0.0
    running_cls = 0.0
    running_obj = 0.0
    total_batches = max(1, len(dataloader))

    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        if use_channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        with get_autocast_context(device, use_amp):
            predictions = model(images)
            loss, loss_items = compute_loss(predictions, targets)

        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_items["loss"]
            running_reg += loss_items["reg_loss"]
            running_cls += loss_items["cls_loss"]
            running_obj += loss_items["obj_loss"]
        else:
            print("Warning: NaN or Inf loss encountered, skipping step.")

        pbar.set_postfix(
            {
                "loss": loss_items["loss"],
                "reg": loss_items["reg_loss"],
                "cls": loss_items["cls_loss"],
                "obj": loss_items["obj_loss"],
                "pos": loss_items["num_pos"],
            }
        )

    return {
        "loss": running_loss / total_batches,
        "reg_loss": running_reg / total_batches,
        "cls_loss": running_cls / total_batches,
        "obj_loss": running_obj / total_batches,
    }


@torch.no_grad()
def validate(model, dataloader, compute_loss, device, use_amp, use_channels_last):
    model.eval()
    running_loss = 0.0
    running_reg = 0.0
    running_cls = 0.0
    running_obj = 0.0
    total_batches = max(1, len(dataloader))

    pbar = tqdm(dataloader, desc="Validation")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        if use_channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with get_autocast_context(device, use_amp):
            predictions = model(images)
            loss, loss_items = compute_loss(predictions, targets)

        running_loss += loss_items["loss"]
        running_reg += loss_items["reg_loss"]
        running_cls += loss_items["cls_loss"]
        running_obj += loss_items["obj_loss"]
        pbar.set_postfix({"loss": loss_items["loss"]})

    return {
        "loss": running_loss / total_batches,
        "reg_loss": running_reg / total_batches,
        "cls_loss": running_cls / total_batches,
        "obj_loss": running_obj / total_batches,
    }


def load_evaluation_module(val_data_path):
    repo_root = Path(__file__).resolve().parent
    dataset_root = Path(val_data_path).resolve().parents[1]
    candidates = [
        dataset_root / "tools" / "evaluate_predictions.py",
        repo_root / "public" / "tools" / "evaluate_predictions.py",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("evaluate_predictions_local", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, candidate

    return None, None


@torch.no_grad()
def evaluate_map(model, val_data_path, val_image_dir, img_size, device, use_amp, classes, use_channels_last):
    model.eval()
    evaluate_predictions, module_path = load_evaluation_module(val_data_path)
    if evaluate_predictions is None:
        print("Warning: Could not locate evaluate_predictions.py. Skipping mAP evaluation.")
        return 0.0

    print(f"\n--- Running mAP Evaluation ({module_path}) ---")
    with open(val_data_path, "r", encoding="utf-8") as handle:
        gt_data = json.load(handle)

    _, image_info = evaluate_predictions.validate_ground_truth(gt_data)
    predictions_list = []

    for img_id in tqdm(list(image_info.keys()), desc="mAP Inference"):
        img_filename = os.path.basename(image_info[img_id]["file_name"])
        img_path = os.path.join(val_image_dir, img_filename)

        img = cv2.imread(img_path)
        if img is None:
            predictions_list.append({"image_id": img_id, "boxes": []})
            continue

        h_orig, w_orig = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std

        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0).to(device, non_blocking=True)
        if use_channels_last:
            img_tensor = img_tensor.contiguous(memory_format=torch.channels_last)

        with get_autocast_context(device, use_amp):
            preds = model(img_tensor)

        strides = {"p3": 8, "p4": 16, "p5": 32}
        all_bboxes = []
        all_obj_scores = []
        all_cls_probs = []

        for level, stride in strides.items():
            cls_pred, reg_pred, obj_pred = preds[level]
            bboxes, obj_scores, cls_probs = decode_predictions(reg_pred, obj_pred, cls_pred, stride, device)
            all_bboxes.append(bboxes)
            all_obj_scores.append(obj_scores)
            all_cls_probs.append(cls_probs)

        detections = non_max_suppression(
            torch.cat(all_bboxes, dim=1),
            torch.cat(all_obj_scores, dim=1),
            torch.cat(all_cls_probs, dim=1),
            conf_thres=0.05,
            iou_thres=0.5,
        )

        boxes_out = []
        for det in detections[0]:
            xmin = float(det[0]) * (w_orig / img_size)
            ymin = float(det[1]) * (h_orig / img_size)
            xmax = float(det[2]) * (w_orig / img_size)
            ymax = float(det[3]) * (h_orig / img_size)
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
                        "confidence": conf,
                        "bbox": [xmin, ymin, xmax, ymax],
                    }
                )

        predictions_list.append({"image_id": img_id, "boxes": boxes_out})

    try:
        normalized_preds = evaluate_predictions.normalize_predictions(
            predictions_list,
            classes=classes,
            image_info=image_info,
            max_detections_per_image=100,
            require_complete=True,
        )
        results = evaluate_predictions.evaluate(
            ground_truth=gt_data,
            predictions=normalized_preds,
            classes=classes,
            iou_threshold=0.5,
        )
        current_map = results.get("mAP@0.5", 0.0)
        print(
            f"\n=================== mAP@0.5: {current_map:.4f} "
            f"(Points: {results.get('performance_points', 0)}) ==================="
        )
        return current_map
    except Exception as exc:
        print(f"Error during evaluate_predictions evaluation: {exc}")
        return 0.0


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = detect_device()
    configure_torch_runtime(device)
    use_amp = device.type == "cuda"
    use_channels_last = device.type == "cuda"

    args.batch_size = resolve_batch_size(args.batch_size, device)
    args.num_workers = resolve_num_workers(args.num_workers, device)

    print(f"Using device: {device} ({get_gpu_name(device)})")
    print(
        f"Training config: batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"img_size={args.img_size}, amp={use_amp}, channels_last={use_channels_last}"
    )

    print("Loading datasets...")
    train_dataset = ObjectDetectionDataset(
        json_path=args.train_data,
        img_dir=args.image_dir,
        img_size=args.img_size,
        augment=True,
        mosaic_prob=args.mosaic_prob,
        mixup_prob=args.mixup_prob,
    )
    val_dataset = ObjectDetectionDataset(
        json_path=args.val_data,
        img_dir=args.val_image_dir,
        img_size=args.img_size,
        augment=False,
    )

    dataloader_kwargs = get_dataloader_kwargs(args.num_workers, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )

    num_classes = len(train_dataset.classes)
    print(f"Initializing Detector model with {num_classes} classes...")
    model = Detector(num_classes=num_classes, pretrained=True)
    if use_channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    compute_loss = ComputeLoss(num_classes=num_classes, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = min(3, max(1, args.epochs // 10 + 1))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        denom = max(1, args.epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / denom
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = get_grad_scaler(use_amp)

    start_epoch = 0
    best_val_loss = float("inf")
    best_map = 0.0
    checkpoint_path = resolve_checkpoint_path(args.weights, args.checkpoint_dir, prefer_last=args.resume)

    if checkpoint_path is not None:
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        start_epoch, best_val_loss, best_map = maybe_load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_path=checkpoint_path,
            device="cpu",
            resume=args.resume,
        )
        if args.resume:
            print(f"Resuming training from epoch {start_epoch + 1}.")
        else:
            print("Loaded model weights for warm start / fine-tuning.")
    elif args.weights:
        print(f"Warning: could not find weights from '{args.weights}'. Starting from pretrained backbone only.")

    patience_counter = 0
    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")
        if args.close_mosaic_epochs > 0 and epoch == max(start_epoch, args.epochs - args.close_mosaic_epochs):
            train_dataset.close_advanced_augment()
            print("Disabled mosaic/mixup for final fine-tuning epochs.")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            compute_loss=compute_loss,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
        )
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            compute_loss=compute_loss,
            device=device,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
        )
        scheduler.step()

        print(
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"[Reg: {train_metrics['reg_loss']:.4f}, Cls: {train_metrics['cls_loss']:.4f}, Obj: {train_metrics['obj_loss']:.4f}]"
        )
        print(
            f"Val Loss:   {val_metrics['loss']:.4f} "
            f"[Reg: {val_metrics['reg_loss']:.4f}, Cls: {val_metrics['cls_loss']:.4f}, Obj: {val_metrics['obj_loss']:.4f}]"
        )

        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            best_map=best_map,
            classes=train_dataset.classes,
            args=args,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            checkpoint["best_val_loss"] = best_val_loss
            checkpoint["best_map"] = best_map

            best_loss_path = os.path.join(args.checkpoint_dir, CHECKPOINT_FILES["best_loss"])
            save_checkpoint(best_loss_path, checkpoint)
            print(f"New best loss model saved to {best_loss_path} (Val Loss: {best_val_loss:.4f})")

            best_path = os.path.join(args.checkpoint_dir, CHECKPOINT_FILES["best"])
            if not os.path.exists(best_path):
                save_checkpoint(best_path, checkpoint)
        else:
            patience_counter += 1
            print(f"Early Stopping counter: {patience_counter}/{args.patience} epochs without improvement.")

        should_run_map = args.map_interval > 0 and (((epoch + 1) % args.map_interval) == 0 or (epoch + 1) == args.epochs)
        if should_run_map:
            current_map = evaluate_map(
                model=model,
                val_data_path=args.val_data,
                val_image_dir=args.val_image_dir,
                img_size=args.img_size,
                device=device,
                use_amp=use_amp,
                classes=val_dataset.classes,
                use_channels_last=use_channels_last,
            )
            if current_map >= best_map:
                best_map = current_map
                checkpoint["best_val_loss"] = best_val_loss
                checkpoint["best_map"] = best_map
                best_path = os.path.join(args.checkpoint_dir, CHECKPOINT_FILES["best"])
                save_checkpoint(best_path, checkpoint)
                print(f"New best mAP model saved to {best_path} (mAP@0.5: {best_map:.4f})")

        checkpoint["best_val_loss"] = best_val_loss
        checkpoint["best_map"] = best_map
        last_path = os.path.join(args.checkpoint_dir, CHECKPOINT_FILES["last"])
        save_checkpoint(last_path, checkpoint)

        if patience_counter >= args.patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1} because validation loss did not improve for "
                f"{args.patience} consecutive epochs."
            )
            if args.map_interval > 0 and not should_run_map:
                current_map = evaluate_map(
                    model=model,
                    val_data_path=args.val_data,
                    val_image_dir=args.val_image_dir,
                    img_size=args.img_size,
                    device=device,
                    use_amp=use_amp,
                    classes=val_dataset.classes,
                    use_channels_last=use_channels_last,
                )
                if current_map >= best_map:
                    best_map = current_map
                    checkpoint["best_val_loss"] = best_val_loss
                    checkpoint["best_map"] = best_map
                    best_path = os.path.join(args.checkpoint_dir, CHECKPOINT_FILES["best"])
                    save_checkpoint(best_path, checkpoint)
                    print(f"New best mAP model saved to {best_path} (mAP@0.5: {best_map:.4f})")
            checkpoint["best_val_loss"] = best_val_loss
            checkpoint["best_map"] = best_map
            save_checkpoint(os.path.join(args.checkpoint_dir, CHECKPOINT_FILES["last"]), checkpoint)
            break

    print("\nTraining completed!")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")
    print(f"Best mAP@0.5 achieved: {best_map:.4f}")


if __name__ == "__main__":
    main()
