# Mô hình Phát hiện Đối tượng Tùy chỉnh (Anchor-free Object Detection)

Dự án này cài đặt một mô hình phát hiện đối tượng (object detection) hoàn toàn từ đầu (không sử dụng các framework có sẵn như YOLOv5/v8, Detectron2 hay torchvision detection APIs) nhằm giải quyết bài toán phát hiện 5 lớp đối tượng: `person`, `car`, `dog`, `cat`, và `chair`.

Mô hình sử dụng backbone **ConvNeXt-Tiny** kết hợp với **PANet** và **Decoupled Head (YOLOX-style)**, huấn luyện theo cơ chế **Anchor-free** để đạt được mAP tối đa.

---

## 1. Thiết kế Kiến trúc Mô hình
*   **Backbone:** ConvNeXt-Tiny (Sử dụng pre-trained weights từ ImageNet) để trích xuất đặc trưng đa quy mô tại Stride 8 (P3), Stride 16 (P4), và Stride 32 (P5).
*   **Neck:** PANet (Path Aggregation Network) giúp tổng hợp thông tin ngữ nghĩa mạnh từ trên xuống (Top-down) và bảo toàn thông tin định vị chính xác từ dưới lên (Bottom-up). Số kênh đặc trưng của Neck được đặt là `256` để tăng khả năng biểu diễn thông tin.
*   **Head:** Decoupled Head (YOLOX-style) giúp tách biệt hai nhiệm vụ phân lớp (Classification - dự đoán 5 lớp) và định vị (Regression - tọa độ hộp và điểm có đối tượng Objectness).
*   **Target Assignment (Gán mẫu):** Cơ chế Anchor-free Center-sampling gán đối tượng thực tế (GT box) vào grid cell gần tâm nhất trên feature level tương ứng theo dải kích thước đối tượng (Scale assignment), đồng thời lấy thêm các cell lân cận để tăng độ ổn định.
*   **Hàm mất mát (Loss Function):** 
    *   *Regression Loss:* Complete IoU (CIoU) Loss nhằm trực tiếp tối ưu hóa độ chồng chập, khoảng cách tâm và tỷ lệ khung hình.
    *   *Classification Loss:* BCE Loss với logits.
    *   *Objectness Loss:* BCE Loss với logits (đích target của các cell dương được gán bằng IoU thực tế giữa predicted box và GT box để cải thiện độ chính xác tin cậy).

---

## 2. Cấu trúc mã nguồn
```
├── models/
│   ├── detector.py       # Kiến trúc mô hình ConvNeXt-Tiny + PANet + Decoupled Head & giải mã tọa độ
│   └── checkpoint/       # Thư mục lưu trữ các checkpoint weights (.pth) của model
├── utils/
│   ├── dataset.py        # Dataloader, tiền xử lý và tăng cường dữ liệu (Mosaic, Mixup, Albumentations)
│   ├── loss.py           # Công thức tính hàm loss (CIoU + BCE) và cơ chế Center-matching
│   ├── nms.py            # Hàm tính toán IoU và Class-wise NMS tự viết hoàn toàn từ đầu
│   └── config.py         # Chứa hàm cấu hình runtime, thiết bị, và các thiết lập mặc định
├── datasets/
│   ├── annotations/      # Chứa các file annotation JSON (train.json, val.json,...)
│   ├── classes.json      # File định nghĩa 5 class của dự án
│   ├── tools/
│   │   ├── convert.py    # Script chuyển đổi file JSON dự đoán sang định dạng CSV nộp bài
│   │   └── evaluate_predictions.py # Công cụ tự chấm điểm mAP cục bộ
│   ├── train/            # Thư mục ảnh huấn luyện
│   ├── val/              # Thư mục ảnh đánh giá
│   └── test/             # Thư mục ảnh kiểm thử
├── train.py              # Kịch bản huấn luyện chính (AMP, AdamW, Cosine Warmup, checkpointing)
├── predict.py            # Kịch bản suy luận chính (đọc ảnh -> dự đoán -> NMS -> lưu file JSON)
├── requirements.txt      # Các thư viện phụ thuộc
└── README.md             # Hướng dẫn sử dụng (tệp này)
```

---

## 3. Hướng dẫn thiết lập môi trường

### Bước 1: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 2: Cài đặt PyTorch hỗ trợ GPU (CUDA 12.1/11.8)
Nếu bạn huấn luyện trên máy cục bộ có GPU hoặc trên Google Colab / Kaggle, hãy chắc chắn PyTorch được cài đặt đúng phiên bản hỗ trợ GPU để tăng tốc độ huấn luyện:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

---

## 4. Hướng dẫn Huấn luyện (Training)

Chạy lệnh huấn luyện dưới đây để bắt đầu huấn luyện mô hình:
```bash
python train.py \
  --train_data datasets/annotations/train.json \
  --val_data datasets/annotations/val.json \
  --image_dir datasets/train/images \
  --val_image_dir datasets/val/images \
  --checkpoint_dir models/checkpoint \
  --epochs 30 \
  --batch_size 8 \
  --img_size 512
```

### Tiếp tục huấn luyện bị gián đoạn (Resume training)
Nếu việc huấn luyện bị mất kết nối internet (như trên Colab) hoặc bị dừng giữa chừng, bạn có thể khôi phục lại trạng thái huấn luyện bằng cách thêm flag `--resume`:
```bash
python train.py \
  --train_data datasets/annotations/train.json \
  --val_data datasets/annotations/val.json \
  --image_dir datasets/train/images \
  --val_image_dir datasets/val/images \
  --checkpoint_dir models/checkpoint \
  --resume
```
*(Hệ thống sẽ tự động tìm file `last.pth` trong thư mục checkpoint để khôi phục toàn bộ Model Weights, Optimizer, Scheduler, Scaler và chạy tiếp từ epoch tiếp theo).*

---

## 5. Hướng dẫn Suy luận (Inference / Predict)

Chạy suy luận trên tập ảnh kiểm tra và xuất kết quả dự đoán ra file JSON:
```bash
python predict.py \
  --image_dir datasets/test/images \
  --output test_predictions.json \
  --weights models/checkpoint/best_loss.pth \
  --conf_thres 0.05 \
  --iou_thres 0.5
```
*Lưu ý:*
*   `--weights`: Đường dẫn đến file checkpoint tốt nhất thu được khi train (`best_loss.pth` hoặc `best.pth`).
*   `--conf_thres`: Ngưỡng độ tin cậy để lọc box (mặc định: `0.05`).
*   `--iou_thres`: Ngưỡng IoU lọc trùng (NMS) (mặc định: `0.5`).

---

## 6. Chuyển đổi định dạng nộp bài (JSON sang CSV)

Để chuyển đổi file kết quả dự đoán JSON thành định dạng CSV (submission format) có 2 cột `image_id` và `bounding_boxes`, hãy chạy:
```bash
python datasets/tools/convert.py \
  --input test_predictions.json \
  --output submission.csv \
  --classes datasets/classes.json
```

---

## 7. Hướng dẫn tự chấm điểm (Self-Evaluation / Scoring)

Bạn có thể tự tính toán điểm số (mAP@0.5, Precision, Recall) trên tập kiểm định để theo dõi hiệu năng cục bộ:
```bash
python datasets/tools/evaluate_predictions.py \
  --ground_truth datasets/annotations/val.json \
  --predictions test_predictions.json \
  --output score.json
```
Lệnh này sẽ tự động sinh ra file `score.json` lưu điểm số mAP chi tiết của từng class và điểm tổng của mô hình.

---

## 8. Hướng dẫn nộp bài (Submission & Docker)

Để phục vụ quá trình chấm bài bằng Docker của giáo viên, cấu trúc thư mục nộp bài cần được tổ chức như sau:

### Cấu trúc thư mục nộp bài chuẩn:
```
├── Dockerfile                  # Đặt ngang hàng với thư mục code nộp bài
└── my_submission/              # Thư mục chứa toàn bộ dự án của bạn (đổi tên từ thư mục 'final')
    ├── predict.py
    ├── train.py
    ├── requirements.txt
    ├── models/
    │   ├── best.pth            # [CỰC KỲ QUAN TRỌNG] Phải là file weight tốt nhất của bạn
    │   └── detector.py
    ├── utils/
    └── datasets/
```

> [!WARNING]
> **LƯU Ý CỰC KỲ QUAN TRỌNG VỀ WEIGHTS:**
> Lệnh chạy Docker của giáo viên **không truyền tham số `--weights`**, điều này có nghĩa là script `predict.py` sẽ chạy bằng đường dẫn mặc định là `./models/best.pth`.
> 
> Do đó, trước khi nộp bài, bạn **bắt buộc phải copy** tệp trọng số tốt nhất của mình (ví dụ từ `models/checkpoint/best_loss.pth` hoặc checkpoint tốt nhất mà bạn vừa huấn luyện xong) và lưu đè/đặt tên thành **`models/best.pth`**. Nếu không, hệ thống chấm bài sẽ chạy bằng mô hình cũ.

### Các lệnh kiểm tra build và chạy Docker:

1. **Đứng tại thư mục cha (thư mục chứa `Dockerfile` và `my_submission/`) để build image:**
   ```bash
   docker build -t object-detection-exam:2026 .
   ```

2. **Di chuyển vào thư mục nộp bài:**
   ```bash
   cd my_submission
   ```

3. **Chạy container để suy luận trên tập đánh giá:**
   ```bash
   # Tạo thư mục đầu ra
   mkdir -p grading_outputs

   # Chạy suy luận qua Docker (sử dụng GPU)
   docker run --rm --gpus all \
     -v "$PWD/public/val/images:/exam/val_images:ro" \
     -v "$PWD:/workspace" \
     -v "$PWD/grading_outputs:/exam/outputs" \
     object-detection-exam:2026 \
     python predict.py \
       --image_dir /exam/val_images \
       --output /exam/outputs/val_predictions.json
   ```

4. **Tự chấm điểm kết quả từ Docker:**
   ```bash
   python public/tools/evaluate_predictions.py \
     --ground_truth public/annotations/val.json \
     --predictions grading_outputs/val_predictions.json \
     --output grading_outputs/val_score.json
   ```

