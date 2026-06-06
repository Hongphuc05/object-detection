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
│   └── detector.py       # Kiến trúc mô hình ConvNeXt-Tiny + PANet + Decoupled Head & giải mã tọa độ
├── utils/
│   ├── dataset.py        # Dataloader, tiền xử lý và tăng cường dữ liệu (Mosaic, Mixup, Albumentations)
│   ├── loss.py           # Công thức tính hàm loss (CIoU + BCE) và cơ chế Center-matching
│   └── nms.py            # Hàm tính toán IoU và Class-wise NMS tự viết hoàn toàn từ đầu
├── train.py              # Kịch bản huấn luyện chính (AMP, AdamW, Cosine Warmup, checkpointing)
├── predict.py            # Kịch bản suy luận chính (đọc ảnh -> dự đoán -> NMS -> lưu file JSON)
├── requirements.txt      # Các thư viện phụ thuộc
└── README.md             # Hướng dẫn này
```

---

## 3. Hướng dẫn thiết lập môi trường

### Bước 1: Chuẩn bị môi trường Python
Khuyên dùng Python 3.12 (hoặc Python 3.8+).

### Bước 2: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 3: Cài đặt PyTorch có hỗ trợ GPU (CUDA 12.1)
Để quá trình huấn luyện diễn ra nhanh chóng trên GPU (như NVIDIA GTX 1650 Ti cục bộ hoặc T4 trên Colab/Kaggle), hãy cài đặt bản PyTorch hỗ trợ GPU bằng lệnh:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```
*(Nếu chạy trên các nền tảng đám mây như Google Colab hoặc Kaggle, GPU đã được cấu hình sẵn trong môi trường).*

---

## 4. Hướng dẫn Huấn luyện (Training)

Chạy lệnh huấn luyện bắt buộc sau để bắt đầu huấn luyện mô hình:
```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```
*Lưu ý:*
*   Trọng số mô hình có Validation Loss tốt nhất sẽ tự động được lưu vào tệp `./models/best.pth`.
*   Trọng số của epoch cuối cùng được lưu vào `./models/last.pth`.
*   Bạn có thể truyền thêm các tham số tùy chọn như `--epochs` (mặc định: 30), `--batch_size` (mặc định: 8), `--img_size` (mặc định: 512) và `--lr` (mặc định: 0.001) để tối ưu thêm.

---

## 5. Hướng dẫn Suy luận (Inference / Predict)

Để chạy suy luận trên tập ảnh kiểm tra và xuất kết quả theo đúng định dạng yêu cầu, sử dụng lệnh bắt buộc:
```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```
*Lưu ý:*
*   Kịch bản suy luận mặc định sẽ tải trọng số mô hình tốt nhất từ `./models/best.pth`.
*   Các tham số ngưỡng tin cậy `--conf_thres` (mặc định: 0.05) và ngưỡng NMS `--iou_thres` (mặc định: 0.5) có thể điều chỉnh để tìm điểm cân bằng Recall/Precision tối ưu nhất khi chấm mAP.
*   Tệp đầu ra `predictions.json` sẽ được định dạng chuẩn JSON mảng chứa thông tin đối tượng và hộp bao tương tự tệp kiểm tra mẫu.
