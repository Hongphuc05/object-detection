import os
import csv
import json
import cv2
from tqdm import tqdm

def process_csv(csv_path, images_dir, labels_dir, output_json_path):
    print(f"Processing {csv_path}...")
    
    # Target classes mapping
    # 6: car, 7: cat, 8: chair, 11: dog, 14: person
    class_mapping = {
        6: "car",
        7: "cat",
        8: "chair",
        11: "dog",
        14: "person"
    }
    
    target_classes = ["person", "car", "dog", "cat", "chair"]
    
    images_list = []
    annotations_list = []
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = list(reader)
        
    # Check if header exists and skip it
    if len(lines) > 0 and lines[0][0] == "img" and lines[0][1] == "label":
        lines = lines[1:]
        
    for row in tqdm(lines, desc="Processing images"):
        if len(row) < 2:
            continue
        img_name, label_name = row[0], row[1]
        
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, label_name)
        
        if not os.path.exists(img_path):
            continue
            
        # Read image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w, _ = img.shape
        
        # Read labels if exists
        valid_annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_idx = int(parts[0])
                    
                    if class_idx in class_mapping:
                        class_name = class_mapping[class_idx]
                        
                        # YOLO format: x_center, y_center, w, h normalized
                        x_c = float(parts[1])
                        y_c = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        
                        # Convert to absolute COCO-like bbox [xmin, ymin, xmax, ymax]
                        xmin = (x_c - w / 2) * img_w
                        ymin = (y_c - h / 2) * img_h
                        xmax = (x_c + w / 2) * img_w
                        ymax = (y_c + h / 2) * img_h
                        
                        # Clip boundaries
                        xmin = max(0.0, min(xmin, float(img_w)))
                        ymin = max(0.0, min(ymin, float(img_h)))
                        xmax = max(0.0, min(xmax, float(img_w)))
                        ymax = max(0.0, min(ymax, float(img_h)))
                        
                        # Check validity
                        if xmax > xmin and ymax > ymin:
                            valid_annotations.append({
                                "image_id": img_name,
                                "class": class_name,
                                "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)]
                            })
                            
        # Only include the image if it has at least one valid target annotation
        if len(valid_annotations) > 0:
            images_list.append({
                "id": img_name,
                "file_name": f"pascal/images/{img_name}",
                "width": img_w,
                "height": img_h
            })
            annotations_list.extend(valid_annotations)
            
    # Save JSON structure
    output_data = {
        "classes": target_classes,
        "images": images_list,
        "annotations": annotations_list
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(images_list)} images and {len(annotations_list)} annotations to {output_json_path}\n")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pascal_dir = os.path.join(base_dir, "data", "pascal")
    
    images_dir = os.path.join(pascal_dir, "images")
    labels_dir = os.path.join(pascal_dir, "labels")
    
    train_csv = os.path.join(pascal_dir, "train.csv")
    test_csv = os.path.join(pascal_dir, "test.csv")
    
    annotations_dir = os.path.join(pascal_dir, "annotations")
    train_json = os.path.join(annotations_dir, "train.json")
    val_json = os.path.join(annotations_dir, "val.json")
    
    # Process train and val (test) datasets
    process_csv(train_csv, images_dir, labels_dir, train_json)
    process_csv(test_csv, images_dir, labels_dir, val_json)

if __name__ == "__main__":
    main()
