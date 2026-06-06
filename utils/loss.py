import torch
import torch.nn as nn
import math

def calculate_iou(box1, box2):
    """
    Computes IoU between two tensors of boxes.
    Args:
        box1 (Tensor): [N, 4] boxes (xmin, ymin, xmax, ymax)
        box2 (Tensor): [N, 4] boxes (xmin, ymin, xmax, ymax)
    """
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter + 1e-7
    return inter / union

def calculate_ciou(box1, box2):
    """
    Computes Complete IoU (CIoU) Loss.
    Args:
        box1 (Tensor): Predicted boxes [N, 4] (xmin, ymin, xmax, ymax)
        box2 (Tensor): Ground truth boxes [N, 4] (xmin, ymin, xmax, ymax)
    """
    # 1. IoU
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1]
    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1]
    
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter + 1e-7
    iou = inter / union

    # 2. Distance term
    cx1 = (box1[:, 0] + box1[:, 2]) / 2
    cy1 = (box1[:, 1] + box1[:, 3]) / 2
    cx2 = (box2[:, 0] + box2[:, 2]) / 2
    cy2 = (box2[:, 1] + box2[:, 3]) / 2
    
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing diagonal
    enc_x1 = torch.min(box1[:, 0], box2[:, 0])
    enc_y1 = torch.min(box1[:, 1], box2[:, 1])
    enc_x2 = torch.max(box1[:, 2], box2[:, 2])
    enc_y2 = torch.max(box1[:, 3], box2[:, 3])
    
    diag_dist_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    # 3. Aspect ratio consistency v and trade-off parameter alpha
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + 1e-7)) - torch.atan(w1 / (h1 + 1e-7)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (center_dist_sq / diag_dist_sq + alpha * v)
    return 1.0 - torch.clamp(ciou, min=-1.0, max=1.0)

class ComputeLoss:
    def __init__(self, num_classes=5, device=torch.device('cpu')):
        self.num_classes = num_classes
        self.device = device
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def __call__(self, predictions, targets):
        """
        Computes the multi-task loss for the detector.
        Args:
            predictions (dict): Output from Detector forward pass containing:
                                'p3': (cls3, reg3, obj3) -> stride 8
                                'p4': (cls4, reg4, obj4) -> stride 16
                                'p5': (cls5, reg5, obj5) -> stride 32
            targets (list[dict]): List of length Batch size containing GT boxes and labels.
        """
        total_loss = 0.0
        reg_loss = 0.0
        cls_loss = 0.0
        obj_loss = 0.0

        strides = {'p3': 8, 'p4': 16, 'p5': 32}

        for level, stride in strides.items():
            cls_pred, reg_pred, obj_pred = predictions[level]
            B, _, H, W = reg_pred.shape

            # Target containers
            tgt_obj = torch.zeros((B, 1, H, W), dtype=torch.float32, device=self.device)
            tgt_cls = torch.zeros((B, self.num_classes, H, W), dtype=torch.float32, device=self.device)
            tgt_reg = torch.zeros((B, 4, H, W), dtype=torch.float32, device=self.device)
            tgt_mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=self.device)

            # 1. Target Assignment (Center-sampling matching)
            for b_idx in range(B):
                target = targets[b_idx]
                gt_boxes = target['boxes']  # shape: [num_obj, 4]
                gt_labels = target['labels']  # shape: [num_obj]

                if gt_boxes.shape[0] == 0:
                    continue

                for gt_idx in range(gt_boxes.shape[0]):
                    box = gt_boxes[gt_idx]
                    label = gt_labels[gt_idx]

                    xmin, ymin, xmax, ymax = box
                    w = xmax - xmin
                    h = ymax - ymin
                    max_size = max(w, h)

                    # Scale assignment: Check if this scale belongs to this feature level
                    # Overlapping scale criteria to stabilize boundaries
                    if stride == 8 and max_size >= 96:
                        continue
                    if stride == 16 and (max_size < 32 or max_size >= 256):
                        continue
                    if stride == 32 and max_size < 128:
                        continue

                    # Grid space center coordinates
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    
                    cx_grid = cx / stride
                    cy_grid = cy / stride

                    col = int(cx_grid)
                    row = int(cy_grid)

                    # Keep within grid boundaries
                    col = max(0, min(col, W - 1))
                    row = max(0, min(row, H - 1))

                    # Place center cell as positive match
                    tgt_mask[b_idx, 0, row, col] = 1.0
                    tgt_obj[b_idx, 0, row, col] = 1.0  # Initial target is 1.0
                    tgt_cls[b_idx, label, row, col] = 1.0
                    tgt_reg[b_idx, :, row, col] = box

                    # Match neighbor cells to increase positive sample count
                    # Offset from grid center
                    dx = cx_grid - col
                    dy = cy_grid - row
                    
                    # If center is close to left cell
                    if dx < 0.5 and col > 0:
                        tgt_mask[b_idx, 0, row, col - 1] = 1.0
                        tgt_obj[b_idx, 0, row, col - 1] = 1.0
                        tgt_cls[b_idx, label, row, col - 1] = 1.0
                        tgt_reg[b_idx, :, row, col - 1] = box
                    # If center is close to right cell
                    elif dx >= 0.5 and col < W - 1:
                        tgt_mask[b_idx, 0, row, col + 1] = 1.0
                        tgt_obj[b_idx, 0, row, col + 1] = 1.0
                        tgt_cls[b_idx, label, row, col + 1] = 1.0
                        tgt_reg[b_idx, :, row, col + 1] = box
                        
                    # If center is close to top cell
                    if dy < 0.5 and row > 0:
                        tgt_mask[b_idx, 0, row - 1, col] = 1.0
                        tgt_obj[b_idx, 0, row - 1, col] = 1.0
                        tgt_cls[b_idx, label, row - 1, col] = 1.0
                        tgt_reg[b_idx, :, row - 1, col] = box
                    # If center is close to bottom cell
                    elif dy >= 0.5 and row < H - 1:
                        tgt_mask[b_idx, 0, row + 1, col] = 1.0
                        tgt_obj[b_idx, 0, row + 1, col] = 1.0
                        tgt_cls[b_idx, label, row + 1, col] = 1.0
                        tgt_reg[b_idx, :, row + 1, col] = box

            # 2. Vectorized decoded box regression
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing='ij'
            )
            grid_x = grid_x.view(1, 1, H, W).float()
            grid_y = grid_y.view(1, 1, H, W).float()

            cx_pred = (grid_x + torch.sigmoid(reg_pred[:, 0:1])) * stride
            cy_pred = (grid_y + torch.sigmoid(reg_pred[:, 1:2])) * stride
            w_pred = torch.exp(reg_pred[:, 2:3]) * stride
            h_pred = torch.exp(reg_pred[:, 3:4]) * stride

            xmin_pred = cx_pred - w_pred / 2
            ymin_pred = cy_pred - h_pred / 2
            xmax_pred = cx_pred + w_pred / 2
            ymax_pred = cy_pred + h_pred / 2
            
            decoded_bboxes = torch.cat([xmin_pred, ymin_pred, xmax_pred, ymax_pred], dim=1)  # [B, 4, H, W]

            # 3. Calculate Losses
            pos_indices = (tgt_mask.squeeze(1) == 1.0)
            num_pos = pos_indices.sum().item()

            if num_pos > 0:
                # Gather positive predictions and targets
                pos_pred_boxes = decoded_bboxes.permute(0, 2, 3, 1)[pos_indices]  # [num_pos, 4]
                pos_tgt_boxes = tgt_reg.permute(0, 2, 3, 1)[pos_indices]  # [num_pos, 4]

                # Regression Loss (CIoU)
                pos_ciou_loss = calculate_ciou(pos_pred_boxes, pos_tgt_boxes)
                reg_loss_level = pos_ciou_loss.mean()
                reg_loss += reg_loss_level

                # Classification Loss (BCE)
                pos_pred_cls = cls_pred.permute(0, 2, 3, 1)[pos_indices]  # [num_pos, num_classes]
                pos_tgt_cls = tgt_cls.permute(0, 2, 3, 1)[pos_indices]  # [num_pos, num_classes]
                cls_loss_level = self.bce_loss(pos_pred_cls, pos_tgt_cls)
                cls_loss += cls_loss_level

                # Align Objectness scores targets to actual IoU of decoded predictions
                with torch.no_grad():
                    pos_ious = calculate_iou(pos_pred_boxes, pos_tgt_boxes)  # [num_pos]
                
                # Assign IoU targets to tgt_obj at positive matching cell indexes
                # We extract positive batch, row, and col coordinates
                b_pos, row_pos, col_pos = torch.where(pos_indices)
                tgt_obj[b_pos, 0, row_pos, col_pos] = pos_ious

            # Objectness Loss (BCE on all cells, positive and negative)
            obj_loss_level = self.bce_loss(obj_pred, tgt_obj)
            obj_loss += obj_loss_level

        # Compute weighted total loss
        # Typical weight values: reg = 5.0, cls = 2.0, obj = 2.0
        total_loss = 5.0 * reg_loss + 2.0 * cls_loss + 2.0 * obj_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'cls_loss': cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            'obj_loss': obj_loss.item() if isinstance(obj_loss, torch.Tensor) else obj_loss,
            'num_pos': num_pos
        }
