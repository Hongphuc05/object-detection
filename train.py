import os
import argparse
import math
import json
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from utils.dataset import ObjectDetectionDataset, collate_fn
from models.detector import Detector, decode_predictions
from utils.loss import ComputeLoss
from utils.nms import non_max_suppression

def get_autocast_context(device, enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled=enabled)

def get_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    else:
        return torch.cuda.amp.GradScaler(enabled=enabled)

def parse_args():
    parser = argparse.ArgumentParser(description="Train custom object detection model from scratch.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train annotations json.")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val annotations json.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to train images directory.")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Path to val images directory.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save model checkpoints.")
    
    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping based on val loss.")
    
    return parser.parse_args()

def train_one_epoch(model, dataloader, optimizer, compute_loss, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    running_reg = 0.0
    running_cls = 0.0
    running_obj = 0.0
    total_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # Mixed precision context
        with get_autocast_context(device, use_amp):
            predictions = model(images)
            loss, loss_items = compute_loss(predictions, targets)
            
        if not torch.isnan(loss) and not torch.isinf(loss):
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss_items['loss']
            running_reg += loss_items['reg_loss']
            running_cls += loss_items['cls_loss']
            running_obj += loss_items['obj_loss']
        else:
            print("Warning: NaN or Inf loss encountered, skipping step.")
            
        pbar.set_postfix({
            'loss': loss_items['loss'],
            'reg': loss_items['reg_loss'],
            'cls': loss_items['cls_loss'],
            'obj': loss_items['obj_loss'],
            'pos': loss_items['num_pos']
        })
        
    return {
        'loss': running_loss / total_batches,
        'reg_loss': running_reg / total_batches,
        'cls_loss': running_cls / total_batches,
        'obj_loss': running_obj / total_batches
    }

@torch.no_grad()
def validate(model, dataloader, compute_loss, device, use_amp):
    model.eval()
    running_loss = 0.0
    running_reg = 0.0
    running_cls = 0.0
    running_obj = 0.0
    total_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc="Validation")
    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with get_autocast_context(device, use_amp):
            predictions = model(images)
            loss, loss_items = compute_loss(predictions, targets)
            
        running_loss += loss_items['loss']
        running_reg += loss_items['reg_loss']
        running_cls += loss_items['cls_loss']
        running_obj += loss_items['obj_loss']
        
        pbar.set_postfix({'loss': loss_items['loss']})
        
    return {
        'loss': running_loss / total_batches,
        'reg_loss': running_reg / total_batches,
        'cls_loss': running_cls / total_batches,
        'obj_loss': running_obj / total_batches
    }

@torch.no_grad()
def evaluate_map(model, val_data_path, val_image_dir, img_size, device, use_amp, classes):
    model.eval()
    
    try:
        import sys
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        tools_dir = os.path.join(workspace_dir, "data", "public", "tools")
        if tools_dir not in sys.path:
            sys.path.append(tools_dir)
        # pyrefly: ignore [missing-import]
        import evaluate_predictions
    except Exception as e:
        print(f"Warning: Could not import evaluate_predictions.py: {e}")
        return 0.0

    print("\n--- Running mAP Evaluation (evaluate_predictions.py) ---")
    with open(val_data_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    _, image_info = evaluate_predictions.validate_ground_truth(gt_data)
    
    predictions_list = []
    image_ids = list(image_info.keys())
    
    for img_id in tqdm(image_ids, desc="mAP Inference"):
        img_filename = os.path.basename(image_info[img_id]['file_name'])
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
        
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        with get_autocast_context(device, use_amp):
            preds = model(img_tensor)
            
        strides = {'p3': 8, 'p4': 16, 'p5': 32}
        all_bboxes = []
        all_obj_scores = []
        all_cls_probs = []
        
        for level, stride in strides.items():
            cls_pred, reg_pred, obj_pred = preds[level]
            bboxes, obj_scores, cls_probs = decode_predictions(
                reg_pred, obj_pred, cls_pred, stride, device
            )
            all_bboxes.append(bboxes)
            all_obj_scores.append(obj_scores)
            all_cls_probs.append(cls_probs)
            
        concat_bboxes = torch.cat(all_bboxes, dim=1)
        concat_obj_scores = torch.cat(all_obj_scores, dim=1)
        concat_cls_probs = torch.cat(all_cls_probs, dim=1)
        
        detections = non_max_suppression(
            concat_bboxes, concat_obj_scores, concat_cls_probs,
            conf_thres=0.05, iou_thres=0.5
        )
        
        img_dets = detections[0]
        boxes_out = []
        if img_dets.shape[0] > 0:
            scale_w = w_orig / img_size
            scale_h = h_orig / img_size
            for det in img_dets:
                xmin = float(det[0]) * scale_w
                ymin = float(det[1]) * scale_h
                xmax = float(det[2]) * scale_w
                ymax = float(det[3]) * scale_h
                conf = float(det[4])
                cls_id = int(det[5])
                
                xmin = max(0.0, min(xmin, w_orig))
                ymin = max(0.0, min(ymin, h_orig))
                xmax = max(0.0, min(xmax, w_orig))
                ymax = max(0.0, min(ymax, h_orig))
                
                if xmax > xmin and ymax > ymin:
                    boxes_out.append({
                        "class": classes[cls_id],
                        "confidence": conf,
                        "bbox": [xmin, ymin, xmax, ymax]
                    })
        predictions_list.append({
            "image_id": img_id,
            "boxes": boxes_out
        })
        
    try:
        normalized_preds = evaluate_predictions.normalize_predictions(
            predictions_list,
            classes=classes,
            image_info=image_info,
            max_detections_per_image=100,
            require_complete=True
        )
        
        results = evaluate_predictions.evaluate(
            ground_truth=gt_data,
            predictions=normalized_preds,
            classes=classes,
            iou_threshold=0.5
        )
        
        mAP = results.get("mAP@0.5", 0.0)
        print(f"\n=================== mAP@0.5: {mAP:.4f} (Points: {results.get('performance_points', 0)}) ===================")
        return mAP
    except Exception as e:
        print(f"Error during evaluate_predictions evaluation: {e}")
        return 0.0

def main():
    args = parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device and AMP availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    print(f"Using device: {device} (AMP Enabled: {use_amp})")
    
    # Load Datasets
    print("Loading datasets...")
    train_dataset = ObjectDetectionDataset(
        json_path=args.train_data,
        img_dir=args.image_dir,
        img_size=args.img_size,
        augment=True
    )
    val_dataset = ObjectDetectionDataset(
        json_path=args.val_data,
        img_dir=args.val_image_dir,
        img_size=args.img_size,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize detector and loss helper
    num_classes = len(train_dataset.classes)
    print(f"Initializing Detector model with {num_classes} classes...")
    model = Detector(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    compute_loss = ComputeLoss(num_classes=num_classes, device=device)
    
    # Optimizer (AdamW is superior here to control overfitting)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine Annealing with Warmup scheduler
    warmup_epochs = min(3, args.epochs // 10 + 1)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        # Cosine decay portion
        progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = get_grad_scaler(use_amp)
    
    best_val_loss = float('inf')
    best_map = 0.0
    patience_counter = 0
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")
        
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            compute_loss=compute_loss,
            scaler=scaler,
            device=device,
            use_amp=use_amp
        )
        
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            compute_loss=compute_loss,
            device=device,
            use_amp=use_amp
        )
        
        # Step LR scheduler
        scheduler.step()
        
        print(f"Train Loss: {train_metrics['loss']:.4f} [Reg: {train_metrics['reg_loss']:.4f}, Cls: {train_metrics['cls_loss']:.4f}, Obj: {train_metrics['obj_loss']:.4f}]")
        print(f"Val Loss:   {val_metrics['loss']:.4f} [Reg: {val_metrics['reg_loss']:.4f}, Cls: {val_metrics['cls_loss']:.4f}, Obj: {val_metrics['obj_loss']:.4f}]")
        
        # Save checkpoints
        # Save best weights based on validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_loss_path = os.path.join(args.checkpoint_dir, 'best_loss.pth')
            torch.save(model.state_dict(), best_loss_path)
            print(f"New best loss model saved to {best_loss_path} (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Run mAP evaluation every 5 epochs or at the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            current_map = evaluate_map(
                model=model,
                val_data_path=args.val_data,
                val_image_dir=args.val_image_dir,
                img_size=args.img_size,
                device=device,
                use_amp=use_amp,
                classes=val_dataset.classes
            )
            # Save best weights based on mAP
            if current_map > best_map:
                best_map = current_map
                best_path = os.path.join(args.checkpoint_dir, 'best.pth')
                torch.save(model.state_dict(), best_path)
                print(f"New best mAP model saved to {best_path} (mAP@0.5: {best_map:.4f})")
                
        # Also save last epoch weight
        last_path = os.path.join(args.checkpoint_dir, 'last.pth')
        torch.save(model.state_dict(), last_path)

        # Early stopping condition
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered! Training stopped at epoch {epoch + 1} because validation loss did not improve for {args.patience} consecutive epochs.")
            # Run a final mAP evaluation if it hasn't just been run in this epoch
            if (epoch + 1) % 5 != 0:
                current_map = evaluate_map(
                    model=model,
                    val_data_path=args.val_data,
                    val_image_dir=args.val_image_dir,
                    img_size=args.img_size,
                    device=device,
                    use_amp=use_amp,
                    classes=val_dataset.classes
                )
                if current_map > best_map:
                    best_map = current_map
                    best_path = os.path.join(args.checkpoint_dir, 'best.pth')
                    torch.save(model.state_dict(), best_path)
                    print(f"New best mAP model saved to {best_path} (mAP@0.5: {best_map:.4f})")
            break

    print("\nTraining completed!")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")
    print(f"Best mAP@0.5 achieved: {best_map:.4f}")

if __name__ == "__main__":
    main()
