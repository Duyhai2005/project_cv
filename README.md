# Fruit Ripeness Computer Vision

Dự án gồm hai notebook độc lập cho bài toán nhận diện độ chín trái cây:

- `image_processing_and_ml.ipynb`: xử lý ảnh cổ điển, phân đoạn ảnh, trích xuất đặc trưng và phân loại bằng KNN.
- `deep_learning.ipynb`: phát hiện trái cây theo bounding box bằng YOLOv11, Faster R-CNN và SSD300 VGG16.

## Cấu Trúc

```text
project_cv/
+-- image_processing_and_ml.ipynb
+-- deep_learning.ipynb
+-- archive/
|   +-- fruit_ripeness_dataset/archive (1)/dataset/
|       +-- train/<class>/*.jpg
|       +-- test/<class>/*.jpg
+-- deepl_dataset/
    +-- data.yaml
    +-- train/images, train/labels
    +-- valid/images, valid/labels
    +-- test/images, test/labels
```

Hai notebook dùng hai bộ dữ liệu khác nhau:

- Dataset classification trong `archive/.../dataset` có cấu trúc `train/test/<class>`.
- Dataset detection trong `deepl_dataset` có định dạng YOLO, gồm ảnh và file label `.txt`.

## Môi Trường

Khuyến nghị dùng Python 3.12 và GPU CUDA cho notebook deep learning. CPU vẫn chạy được nhưng thời gian train sẽ rất lâu.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install numpy pandas matplotlib seaborn opencv-python pillow scikit-learn scikit-image tqdm pyyaml torch torchvision torchmetrics ultralytics timm
```

Nếu dùng PyTorch với CUDA, cài bản `torch`/`torchvision` phù hợp với driver và CUDA trên máy.

## Dataset

### Classification Dataset

Notebook `image_processing_and_ml.ipynb` tự động tìm dataset local bằng `FRUIT_RIPENESS_DATASET_DIR` hoặc `DATA_DIR`; nếu không có biến môi trường, notebook tìm trong `archive/fruit_ripeness_dataset/archive (1)/dataset`.

Tổng cộng 19,956 ảnh: 16,217 train và 3,739 test.

| Class | Train | Test |
| --- | ---: | ---: |
| freshapples | 1,693 | 395 |
| freshbanana | 1,581 | 381 |
| freshoranges | 1,466 | 388 |
| rottenapples | 2,342 | 601 |
| rottenbanana | 2,224 | 530 |
| rottenoranges | 1,595 | 403 |
| unripe apple | 1,934 | 371 |
| unripe banana | 2,097 | 400 |
| unripe orange | 1,285 | 270 |

### Detection Dataset

Notebook `deep_learning.ipynb` tự động tìm dataset local bằng `DEEPL_DATASET_DIR` hoặc `DATA_DIR`; nếu không có biến môi trường, notebook tìm trong `deepl_dataset`.

Dataset detection được xuất từ Roboflow, license CC BY 4.0, gồm 4 class:

```text
Ripe, Rotten, Unripe, overripe
```

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 9,978 | 9,978 |
| valid | 712 | 712 |
| test | 713 | 713 |

Số lượng object theo class trong file label:

| Split | Ripe | Rotten | Unripe | overripe |
| --- | ---: | ---: | ---: | ---: |
| train | 3,003 | 2,727 | 3,504 | 3,225 |
| valid | 200 | 189 | 253 | 221 |
| test | 229 | 183 | 225 | 241 |

## Notebook Xử Lý Ảnh Và ML Cổ Điển

File: `image_processing_and_ml.ipynb`

Pipeline chính:

1. Đọc dataset classification local và tạo dataframe `image_path`, `label`, `split`.
2. Mã hóa label bằng `LabelEncoder`; tạo `CNNFruitDataset` và `DataLoader` để kiểm tra batch ảnh 224x224.
3. Trực quan mẫu ảnh theo class và các biểu diễn màu RGB, grayscale, HSV, LAB.
4. Tiền xử lý ảnh: crop giữa với `crop_ratio=0.9`, resize về 128x128 hoặc 256x256, Gaussian blur.
5. Phân đoạn ảnh:
   - Otsu threshold trên kênh HSV-S hoặc LAB-a*.
   - Canny edge trên kênh HSV-V hoặc LAB-L.
   - Morphology, tìm contour lớn nhất, tạo mask và bounding box.
6. Trích xuất đặc trưng:
   - Trung bình 2 kênh màu theo mask.
   - Diện tích contour.
   - Mean HOG trên ảnh grayscale 128x128.
7. Chuẩn hóa đặc trưng bằng `StandardScaler`.
8. Train KNN với `GridSearchCV`, tìm `n_neighbors`, `weights`, `metric`.

Kết quả KNN đã lưu trong notebook:

| Phương pháp | K tối ưu | Trọng số | Khoảng cách | Train Acc | Val Acc | Test Acc | Gap |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| OTSU_HSV | 4 | distance | manhattan | 100.00% | 78.81% | 83.82% | 16.18% |
| OTSU_LAB | 3 | distance | manhattan | 100.00% | 79.79% | 82.21% | 17.79% |
| CANNY_HSV | 7 | distance | manhattan | 99.99% | 54.27% | 57.18% | 42.81% |
| CANNY_LAB | 10 | distance | manhattan | 99.99% | 58.71% | 62.48% | 37.52% |

Nhận xét ngắn: Otsu kết hợp đặc trưng màu cho kết quả tốt hơn Canny trong bài toán classification này. Các mô hình KNN có train accuracy gần 100%, nên cần chú ý nguy cơ overfitting.

## Notebook Deep Learning

File: `deep_learning.ipynb`

Pipeline chính:

1. Cài `ultralytics` và `torchmetrics`.
2. Đặt seed, chọn `device`, xác định `WORK_DIR`.
3. Tìm dataset YOLO local, đọc `data.yaml`, tạo `deepl_dataset_local.yaml` với đường dẫn tuyệt đối để Ultralytics train local.
4. Trực quan ground-truth bounding box và thống kê object theo split/class.
5. Train YOLOv11:
   - Weights mặc định: `yolo11n.pt`, hoặc `YOLO_WEIGHTS_PATH` nếu đã có file local.
   - `YOLO_EPOCHS = 30`, `YOLO_IMGSZ = 640`, `YOLO_BATCH = 16`.
6. Train Faster R-CNN:
   - Backbone ResNet50 FPN.
   - Thêm background class cho Torchvision detection, nên label YOLO được cộng 1.
   - `FRCNN_EPOCHS = 10`, `FRCNN_BATCH = 4`.
7. Train SSD300 VGG16:
   - `SSD_EPOCHS = 20`, `SSD_BATCH = 4`.
8. Đánh giá detection bằng IoU matching, NMS, `torchmetrics.MeanAveragePrecision`, confusion matrix và classification report.
9. Tổng hợp metric: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, Mean IoU, thời gian inference và FPS.

Kết quả tổng hợp đã lưu trong notebook:

| Model | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 | Mean IoU | Sec/Image | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv11 | 0.8422 | 0.9362 | 0.8867 | 0.9291 | 0.8771 | 0.9509 | 0.0122 | 81.72 |
| Faster R-CNN | 0.8224 | 0.9180 | 0.8676 | 0.8915 | 0.7787 | 0.9284 | 0.1211 | 8.25 |
| SSD300 VGG16 | 0.7987 | 0.8679 | 0.8319 | 0.8207 | 0.5694 | 0.8439 | 0.0286 | 34.97 |

Notebook cũng có output riêng từ Ultralytics cho YOLO trên test set:

| Metric | Value |
| --- | ---: |
| Precision | 0.9124 |
| Recall | 0.9274 |
| F1-score | 0.9199 |
| mAP@0.5 | 0.9725 |
| mAP@0.5:0.95 | 0.9226 |
| FPS | 171.04 |

Bảng tổng hợp cuối notebook ưu tiên evaluator tự viết để so sánh cùng logic IoU/matching giữa YOLOv11, Faster R-CNN và SSD300.

## Cách Chạy

### Chạy notebook classification

```powershell
$env:FRUIT_RIPENESS_DATASET_DIR="C:\project\project_cv\archive\fruit_ripeness_dataset\archive (1)\dataset"
jupyter notebook image_processing_and_ml.ipynb
```

Chạy từ trên xuống dưới. Notebook sẽ tạo các trực quan dataset, preprocessing, segmentation, HOG, sau đó train và đánh giá KNN.

### Chạy notebook detection

```powershell
$env:DEEPL_DATASET_DIR="C:\project\project_cv\deepl_dataset"
$env:DL_WORK_DIR="C:\project\project_cv"
$env:USE_PRETRAINED_WEIGHTS="1"
jupyter notebook deep_learning.ipynb
```

Khi chạy local, các file sinh ra có thể nằm trong `DL_WORK_DIR`:

- `deepl_dataset_local.yaml`
- `runs/yolo11_fruit_ripeness/`
- `faster_rcnn_best.pth`
- `ssd300_vgg16_best.pth`
- `yolo11n.pt` nếu Ultralytics tải weights về local

Nếu môi trường không có internet hoặc không muốn tải pretrained weights:

```powershell
$env:YOLO_WEIGHTS_PATH="C:\path\to\yolo11n.pt"
$env:USE_PRETRAINED_WEIGHTS="0"
```

## Lưu Ý

- Kết quả trong README được lấy từ output đã lưu trong hai notebook; khi train lại, metric có thể thay đổi theo hardware, version thư viện và seed.
- Accuracy không phải metric chính cho object detection; notebook deep learning ưu tiên mAP, IoU, Precision, Recall và FPS.
- Dataset detection có thể có nhiều object trong một ảnh, vì vậy tổng object theo class lớn hơn số ảnh.
- Notebook classification đang dùng đặc trưng thủ công và KNN; `CNNFruitDataset`/`DataLoader` mới được dùng để kiểm tra dữ liệu, chưa có phần train CNN trong notebook này.
