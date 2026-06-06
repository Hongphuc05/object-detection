import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ObjectDetectionDataset(Dataset):
    def __init__(self, json_path, img_dir, img_size=512, augment=False):
        """
        Args:
            json_path (str): Path to annotations JSON file (train.json or val.json).
            img_dir (str): Path to image directory.
            img_size (int): Target size for images.
            augment (bool): Whether to apply training augmentations.
        """
        self.img_dir = Path(img_dir) if isinstance(img_dir, str) else img_dir
        self.img_size = img_size
        self.augment = augment

        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.classes = data['classes']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Map image_id to image info
        self.image_info = {img['id']: img for img in data['images']}
        self.image_ids = list(self.image_info.keys())

        # Map image_id to annotations
        self.annotations = {img_id: [] for img_id in self.image_ids}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id in self.annotations:
                self.annotations[img_id].append(ann)

        # Standard albumentations transforms
        if self.augment:
            self.transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, 
                                   border_mode=cv2.BORDER_CONSTANT, cval=(114, 114, 114), p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_area=2, min_visibility=0.1))
        else:
            self.transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __len__(self):
        return len(self.image_ids)

    def load_image_and_annotations(self, index):
        """Loads raw image and raw annotations for a given index."""
        img_id = self.image_ids[index]
        img_info = self.image_info[img_id]
        
        # In the train.json, file_name is like 'train/images/img_xxx.jpg'
        # The img_dir provided is d:\Xulyanh\final\data\public\train\images
        # Since the file_name contains 'train/images/', we extract the basename
        img_filename = os.path.basename(img_info['file_name'])
        img_path = os.path.join(self.img_dir, img_filename)
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        anns = self.annotations[img_id]
        bboxes = []
        labels = []
        for ann in anns:
            # bbox coordinates: [xmin, ymin, xmax, ymax]
            bbox = ann['bbox']
            # Safeguard coords to ensure xmax > xmin and ymax > ymin
            xmin, ymin, xmax, ymax = bbox
            if xmax <= xmin or ymax <= ymin:
                continue
            bboxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            labels.append(self.class_to_idx[ann['class']])
            
        return img, bboxes, labels

    def load_normal(self, index):
        """Loads and resizes image and annotations normally."""
        img, bboxes, labels = self.load_image_and_annotations(index)
        h, w, _ = img.shape
        s = self.img_size
        
        # Scale bounding boxes to match target image size
        scaled_bboxes = []
        for box in bboxes:
            xmin = box[0] * (s / w)
            ymin = box[1] * (s / h)
            xmax = box[2] * (s / w)
            ymax = box[3] * (s / h)
            # Clip to image boundaries
            xmin = max(0.0, min(xmin, s))
            ymin = max(0.0, min(ymin, s))
            xmax = max(0.0, min(xmax, s))
            ymax = max(0.0, min(ymax, s))
            if (xmax - xmin) > 2 and (ymax - ymin) > 2:
                scaled_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                # If box is too small, skip it (label will be removed below)
                pass
                
        # Filter class labels accordingly
        valid_labels = [labels[i] for i in range(len(scaled_bboxes))]
        
        resized_img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        return resized_img, scaled_bboxes, valid_labels

    def load_mosaic(self, index):
        """Loads 4 images and combines them into one mosaic image."""
        s = self.img_size
        # Mosaic center coordinates (randomly chosen within a box in the middle)
        xc = int(random.uniform(s // 2, 3 * s // 2))
        yc = int(random.uniform(s // 2, 3 * s // 2))
        
        # Select 3 other random indices
        indices = [index] + [random.randint(0, len(self.image_ids) - 1) for _ in range(3)]
        
        # Create a 2S x 2S combined image filled with background color 114
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        bboxes4 = []
        labels4 = []
        
        for i, idx in enumerate(indices):
            img, bboxes, labels = self.load_image_and_annotations(idx)
            h, w, _ = img.shape
            
            # Resize image to S x S first
            r = s / max(h, w)
            if r != 1:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
            h, w, _ = img.shape
            
            # Define quadrant placement in the 2S x 2S canvas
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - xc), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - yc)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - xc), min(h, y2a - yc)
                
            # Copy image portion to canvas
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            padw = x1a - x1b
            padh = y1a - y1b
            
            # Adjust and offset bounding boxes
            for box, label in zip(bboxes, labels):
                bx1 = box[0] * r + padw
                by1 = box[1] * r + padh
                bx2 = box[2] * r + padw
                by2 = box[3] * r + padh
                
                # Clip bbox to quadrant boundaries
                bx1 = max(bx1, x1a)
                by1 = max(by1, y1a)
                bx2 = min(bx2, x2a)
                by2 = min(by2, y2a)
                
                # Check validity of bbox
                if (bx2 - bx1) > 2 and (by2 - by1) > 2:
                    bboxes4.append([bx1, by1, bx2, by2])
                    labels4.append(label)
                    
        # Crop the 2S x 2S mosaic image to S x S centered at (xc, yc)
        crop_x1 = max(0, min(xc - s // 2, s))
        crop_y1 = max(0, min(yc - s // 2, s))
        crop_x2 = crop_x1 + s
        crop_y2 = crop_y1 + s
        
        crop_img = mosaic_img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Re-offset the bboxes and clip them to [0, S]
        final_bboxes = []
        final_labels = []
        for box, label in zip(bboxes4, labels4):
            bx1 = max(0.0, box[0] - crop_x1)
            by1 = max(0.0, box[1] - crop_y1)
            bx2 = min(float(s), box[2] - crop_x1)
            by2 = min(float(s), box[3] - crop_y1)
            
            if (bx2 - bx1) > 2 and (by2 - by1) > 2:
                final_bboxes.append([bx1, by1, bx2, by2])
                final_labels.append(label)
                
        return crop_img, final_bboxes, final_labels

    def load_mixup(self, index):
        """Applies Mixup between a mosaic image and another image."""
        img1, bboxes1, labels1 = self.load_mosaic(index)
        
        # Select another random index
        idx2 = random.randint(0, len(self.image_ids) - 1)
        # 50% chance to mix with a mosaic, 50% with normal
        if random.random() < 0.5:
            img2, bboxes2, labels2 = self.load_mosaic(idx2)
        else:
            img2, bboxes2, labels2 = self.load_normal(idx2)
            
        # Blend factor lambda
        r = np.random.beta(8.0, 8.0)
        mixed_img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        
        # Combine bounding boxes and labels
        mixed_bboxes = bboxes1 + bboxes2
        mixed_labels = labels1 + labels2
        
        return mixed_img, mixed_bboxes, mixed_labels

    def __getitem__(self, index):
        # Determine whether to apply advanced data augmentation (Mosaic / Mixup)
        if self.augment:
            p = random.random()
            if p < 0.4:
                # 40% chance: Mixup (which internally calls Mosaic)
                img, bboxes, labels = self.load_mixup(index)
            elif p < 0.8:
                # 40% chance: Mosaic
                img, bboxes, labels = self.load_mosaic(index)
            else:
                # 20% chance: Normal resized image
                img, bboxes, labels = self.load_normal(index)
        else:
            img, bboxes, labels = self.load_normal(index)

        # In case we end up with an empty image (no targets at all) or Albumentations fails,
        # fallback to raw values.
        if len(bboxes) == 0:
            # If no objects, provide a dummy box and target that will be ignored or handled
            # (Albumentations requires at least one bbox if bbox_params are active, or we bypass it)
            # To bypass Albumentations error when no bboxes exist:
            # We can use dummy values and then remove them
            bboxes = [[0.0, 0.0, 10.0, 10.0]]
            labels = [0]
            is_dummy = True
        else:
            is_dummy = False

        # Apply Albumentations transform
        transformed = self.transform(image=img, bboxes=bboxes, class_labels=labels)
        img_tensor = transformed['image']
        trans_bboxes = transformed['bboxes']
        trans_labels = transformed['class_labels']

        # Construct target dictionary
        target = {}
        if is_dummy or len(trans_bboxes) == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
        else:
            target['boxes'] = torch.tensor(trans_bboxes, dtype=torch.float32)
            target['labels'] = torch.tensor(trans_labels, dtype=torch.long)

        return img_tensor, target

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
