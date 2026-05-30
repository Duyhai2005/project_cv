# Fruit Ripeness Computer Vision

Du an gom hai notebook doc lap cho bai toan nhan dien do chin trai cay:

- `image_processing_and_ml.ipynb`: xu ly anh co dien, phan doan anh, trich xuat dac trung va phan loai bang KNN.
- `deep_learning.ipynb`: phat hien trai cay theo bounding box bang YOLOv11, Faster R-CNN va SSD300 VGG16.

## Cau Truc

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

Hai notebook dung hai bo du lieu khac nhau:

- Dataset classification trong `archive/.../dataset` co cau truc `train/test/<class>`.
- Dataset detection trong `deepl_dataset` co dinh dang YOLO, gom anh va file label `.txt`.

## Moi Truong

Khuyen nghi dung Python 3.12 va GPU CUDA cho notebook deep learning. CPU van chay duoc nhung thoi gian train se rat lau.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install numpy pandas matplotlib seaborn opencv-python pillow scikit-learn scikit-image tqdm pyyaml torch torchvision torchmetrics ultralytics timm
```

Neu dung PyTorch voi CUDA, cai ban `torch`/`torchvision` phu hop voi driver va CUDA tren may.

## Dataset

### Classification Dataset

Notebook `image_processing_and_ml.ipynb` tu dong tim dataset local bang `FRUIT_RIPENESS_DATASET_DIR` hoac `DATA_DIR`; neu khong co bien moi truong, notebook tim trong `archive/fruit_ripeness_dataset/archive (1)/dataset`.

Tong cong 19,956 anh: 16,217 train va 3,739 test.

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

Notebook `deep_learning.ipynb` tu dong tim dataset local bang `DEEPL_DATASET_DIR` hoac `DATA_DIR`; neu khong co bien moi truong, notebook tim trong `deepl_dataset`.

Dataset detection duoc xuat tu Roboflow, license CC BY 4.0, gom 4 class:

```text
Ripe, Rotten, Unripe, overripe
```

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 9,978 | 9,978 |
| valid | 712 | 712 |
| test | 713 | 713 |

So luong object theo class trong file label:

| Split | Ripe | Rotten | Unripe | overripe |
| --- | ---: | ---: | ---: | ---: |
| train | 3,003 | 2,727 | 3,504 | 3,225 |
| valid | 200 | 189 | 253 | 221 |
| test | 229 | 183 | 225 | 241 |

## Notebook Xu Ly Anh Va ML Co Dien

File: `image_processing_and_ml.ipynb`

Pipeline chinh:

1. Doc dataset classification local va tao dataframe `image_path`, `label`, `split`.
2. Ma hoa label bang `LabelEncoder`; tao `CNNFruitDataset` va `DataLoader` de kiem tra batch anh 224x224.
3. Truc quan mau anh theo class va cac bieu dien mau RGB, grayscale, HSV, LAB.
4. Tien xu ly anh: crop giua voi `crop_ratio=0.9`, resize ve 128x128 hoac 256x256, Gaussian blur.
5. Phan doan anh:
   - Otsu threshold tren kenh HSV-S hoac LAB-a*.
   - Canny edge tren kenh HSV-V hoac LAB-L.
   - Morphology, tim contour lon nhat, tao mask va bounding box.
6. Trich xuat dac trung:
   - Trung binh 2 kenh mau theo mask.
   - Dien tich contour.
   - Mean HOG tren anh grayscale 128x128.
7. Chuan hoa dac trung bang `StandardScaler`.
8. Train KNN voi `GridSearchCV`, tim `n_neighbors`, `weights`, `metric`.

Ket qua KNN da luu trong notebook:

| Phuong phap | K toi uu | Trong so | Khoang cach | Train Acc | Val Acc | Test Acc | Gap |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| OTSU_HSV | 4 | distance | manhattan | 100.00% | 78.81% | 83.82% | 16.18% |
| OTSU_LAB | 3 | distance | manhattan | 100.00% | 79.79% | 82.21% | 17.79% |
| CANNY_HSV | 7 | distance | manhattan | 99.99% | 54.27% | 57.18% | 42.81% |
| CANNY_LAB | 10 | distance | manhattan | 99.99% | 58.71% | 62.48% | 37.52% |

Nhan xet ngan: Otsu ket hop dac trung mau cho ket qua tot hon Canny trong bai toan classification nay. Cac mo hinh KNN co train accuracy gan 100%, nen can chu y nguy co overfitting.

## Notebook Deep Learning

File: `deep_learning.ipynb`

Pipeline chinh:

1. Cai `ultralytics` va `torchmetrics`.
2. Dat seed, chon `device`, xac dinh `WORK_DIR`.
3. Tim dataset YOLO local, doc `data.yaml`, tao `deepl_dataset_local.yaml` voi duong dan tuyet doi de Ultralytics train on local.
4. Truc quan ground-truth bounding box va thong ke object theo split/class.
5. Train YOLOv11:
   - Weights mac dinh: `yolo11n.pt`, hoac `YOLO_WEIGHTS_PATH` neu da co file local.
   - `YOLO_EPOCHS = 30`, `YOLO_IMGSZ = 640`, `YOLO_BATCH = 16`.
6. Train Faster R-CNN:
   - Backbone ResNet50 FPN.
   - Them background class cho Torchvision detection, nen label YOLO duoc cong 1.
   - `FRCNN_EPOCHS = 10`, `FRCNN_BATCH = 4`.
7. Train SSD300 VGG16:
   - `SSD_EPOCHS = 20`, `SSD_BATCH = 4`.
8. Danh gia detection bang IoU matching, NMS, `torchmetrics.MeanAveragePrecision`, confusion matrix va classification report.
9. Tong hop metric: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, Mean IoU, thoi gian inference va FPS.

Ket qua tong hop da luu trong notebook:

| Model | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 | Mean IoU | Sec/Image | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv11 | 0.8422 | 0.9362 | 0.8867 | 0.9291 | 0.8771 | 0.9509 | 0.0122 | 81.72 |
| Faster R-CNN | 0.8224 | 0.9180 | 0.8676 | 0.8915 | 0.7787 | 0.9284 | 0.1211 | 8.25 |
| SSD300 VGG16 | 0.7987 | 0.8679 | 0.8319 | 0.8207 | 0.5694 | 0.8439 | 0.0286 | 34.97 |

Notebook cung co output rieng tu Ultralytics cho YOLO tren test set:

| Metric | Value |
| --- | ---: |
| Precision | 0.9124 |
| Recall | 0.9274 |
| F1-score | 0.9199 |
| mAP@0.5 | 0.9725 |
| mAP@0.5:0.95 | 0.9226 |
| FPS | 171.04 |

Bang tong hop cuoi notebook uu tien evaluator tu viet de so sanh cung logic IoU/matching giua YOLOv11, Faster R-CNN va SSD300.

## Cach Chay

### Chay notebook classification

```powershell
$env:FRUIT_RIPENESS_DATASET_DIR="C:\project\project_cv\archive\fruit_ripeness_dataset\archive (1)\dataset"
jupyter notebook image_processing_and_ml.ipynb
```

Chay tu tren xuong duoi. Notebook se tao cac truc quan dataset, preprocessing, segmentation, HOG, sau do train va danh gia KNN.

### Chay notebook detection

```powershell
$env:DEEPL_DATASET_DIR="C:\project\project_cv\deepl_dataset"
$env:DL_WORK_DIR="C:\project\project_cv"
$env:USE_PRETRAINED_WEIGHTS="1"
jupyter notebook deep_learning.ipynb
```

Khi chay local, cac file sinh ra co the nam trong `DL_WORK_DIR`:

- `deepl_dataset_local.yaml`
- `runs/yolo11_fruit_ripeness/`
- `faster_rcnn_best.pth`
- `ssd300_vgg16_best.pth`
- `yolo11n.pt` neu Ultralytics tai weights ve local

Neu moi truong khong co internet hoac khong muon tai pretrained weights:

```powershell
$env:YOLO_WEIGHTS_PATH="C:\path\to\yolo11n.pt"
$env:USE_PRETRAINED_WEIGHTS="0"
```

## Luu Y

- Ket qua trong README duoc lay tu output da luu trong hai notebook; khi train lai, metric co the thay doi theo hardware, version thu vien va seed.
- Accuracy khong phai metric chinh cho object detection; notebook deep learning uu tien mAP, IoU, Precision, Recall va FPS.
- Dataset detection co the co nhieu object trong mot anh, vi vay tong object theo class lon hon so anh.
- Notebook classification dang dung dac trung thu cong va KNN; `CNNFruitDataset`/`DataLoader` moi duoc dung de kiem tra du lieu, chua co phan train CNN trong notebook nay.
