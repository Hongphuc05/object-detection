import torch

def py_nms(boxes, scores, iou_threshold):
    """
    Vectorized PyTorch implementation of Non-Maximum Suppression (NMS).
    
    Args:
        boxes (Tensor): [N, 4] bounding boxes in Pascal VOC format (xmin, ymin, xmax, ymax)
        scores (Tensor): [N] confidence scores of the boxes
        iou_threshold (float): IoU threshold for suppression
        
    Returns:
        keep (Tensor): indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort scores in descending order
    _, order = torch.sort(scores, descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break
            
        i = order[0].item()
        keep.append(i)
        
        # Intersection coordinates
        xx1 = torch.clamp(x1[order[1:]], min=x1[i])
        yy1 = torch.clamp(y1[order[1:]], min=y1[i])
        xx2 = torch.clamp(x2[order[1:]], max=x2[i])
        yy2 = torch.clamp(y2[order[1:]], max=y2[i])
        
        # Intersection area
        inter = torch.clamp(xx2 - xx1, min=0.0) * torch.clamp(yy2 - yy1, min=0.0)
        
        # Union area
        union = areas[i] + areas[order[1:]] - inter + 1e-7
        
        # Intersection over Union
        iou = inter / union
        
        # Keep boxes with IoU less than or equal to threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]
        
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def non_max_suppression(all_bboxes, all_obj_scores, all_cls_probs, conf_thres=0.05, iou_thres=0.5):
    """
    Performs class-wise NMS on the concatenated predictions of all feature levels.
    
    Args:
        all_bboxes (Tensor): Decoded bounding boxes [B, total_cells, 4]
        all_obj_scores (Tensor): Objectness scores [B, total_cells, 1]
        all_cls_probs (Tensor): Class probabilities [B, total_cells, num_classes]
        conf_thres (float): Confidence threshold (score = obj_score * class_prob)
        iou_thres (float): IoU threshold for NMS
        
    Returns:
        detections (list[Tensor]): List of length B. Each tensor has shape [num_dets, 6]:
                                   [xmin, ymin, xmax, ymax, confidence, class_id]
    """
    B = all_bboxes.shape[0]
    detections = []
    
    device = all_bboxes.device
    
    for i in range(B):
        # Calculate final class-specific detection scores
        scores = all_obj_scores[i] * all_cls_probs[i]  # [total_cells, num_classes]
        
        # Get maximum class score and the corresponding class ID
        max_scores, class_ids = torch.max(scores, dim=1)  # [total_cells]
        
        # Filter by confidence threshold
        keep_mask = max_scores > conf_thres
        
        # Move filtered tensors to CPU to avoid extremely slow GPU-CPU synchronization inside py_nms loop
        bboxes = all_bboxes[i][keep_mask].cpu()
        confidences = max_scores[keep_mask].cpu()
        cls_ids = class_ids[keep_mask].cpu()
        
        if bboxes.shape[0] == 0:
            detections.append(torch.zeros((0, 6), dtype=torch.float32, device=device))
            continue
            
        # Class-wise NMS trick: offset boxes based on their class ID
        # to suppress overlapping boxes only within the same class.
        offsets = cls_ids.float() * 4096.0
        offset_boxes = bboxes + offsets.unsqueeze(1)
        
        # Run custom NMS on CPU (instantaneous compared to GPU sync)
        keep_indices = py_nms(offset_boxes, confidences, iou_thres)
        
        # Select final boxes, scores, and class IDs and move back to original device
        final_boxes = bboxes[keep_indices].to(device)
        final_scores = confidences[keep_indices].to(device)
        final_cls_ids = cls_ids[keep_indices].float().to(device)
        
        # Stack to form [xmin, ymin, xmax, ymax, confidence, class_id]
        img_dets = torch.cat([final_boxes, final_scores.unsqueeze(1), final_cls_ids.unsqueeze(1)], dim=1)
        detections.append(img_dets)
        
    return detections
