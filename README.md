# Smart India Hackathon (SIH 2026) - Traffic Analytics Prototype

An intelligent, dual-feature traffic analytics pipeline powered by a single shared YOLOv8 detector and ByteTrack tracker, built exclusively on the **India Driving Dataset (IDD Detection)**.

---

## 🚦 System Architecture

```
                          ┌───────────────────────────┐
                          │   Video / CCTV Camera     │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ YOLOv8 (13 IDD Classes)   │
                          │   + ByteTrack Tracker     │
                          └──────┬─────────────┬──────┘
                                 │             │
        ┌────────────────────────┘             └────────────────────────┐
        ▼                                                               ▼
┌─────────────────────────────────┐            ┌─────────────────────────────────┐
│ Module 1: Vehicle Density &     │            │ Module 2: Vulnerable Pedestrian │
│ Bottleneck Identification       │            │ & School-Zone Geofencing        │
│ (src/modules/vehicle_density.py)│            │ (src/modules/pedestrian_alert.py│
└────────────────┬────────────────┘            └────────────────┬────────────────┘
                 │                                              │
                 │  - Vehicle class filter                      │  - 'person' & 'rider' filter
                 │  - Multi-zone occupancy                      │  - OpenStreetMap Overpass API
                 │  - Dwell time & velocity                     │  - Point-in-Polygon Geofence
                 │  - Bottleneck stall flags                    │  - Time-of-day school hours
                 │                                              │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                          ┌───────────────────────────┐
                          │     Unified Live HUD      │
                          │      (src/infer.py)       │
                          └───────────────────────────┘
```

---

## 📁 Project Structure

```
e:\SIH_2026\
├── data/
│   ├── raw/idd/                  # Extracted IDD images & annotations
│   └── processed/
│       ├── images/               # Subsampled images (train / val)
│       └── labels/               # YOLO .txt format annotations (train / val)
├── configs/
│   └── data.yaml                 # 13-class IDD YOLOv8 configuration
├── models/                       # Checkpoints (yolov8n.pt, best.pt)
├── src/
│   ├── train.py                  # Training script with hardware auto-tuning
│   ├── infer.py                  # End-to-end inference and HUD visualizer
│   ├── modules/
│   │   ├── vehicle_density.py    # Module 1: Vehicle counting, dwell & bottleneck
│   │   └── pedestrian_alert.py   # Module 2: Pedestrian/rider school geofence alert
│   └── utils/
│       ├── dataset_conversion.py # IDD to YOLO converter + Stratified Subsampler
│       └── geofence.py           # Overpass API client + offline mock geofences
├── requirements.txt              # Project dependencies
└── README.md
```

---

## 🛠️ Installation

```bash
# 1. Clone or navigate to the project
cd e:\SIH_2026

# 2. Install base dependencies
pip install -r requirements.txt

# 3. For GPU Acceleration (NVIDIA RTX 2050):
# If torch.cuda.is_available() is False, install the CUDA-enabled PyTorch build:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📊 Dataset Preparation: Stratified Subsampling

The raw IDD Detection dataset contains ~40,000 frames (~22.8 GB). Training on all 40k images is unnecessary and inefficient for a hackathon timeline.

Our `dataset_conversion.py` implements **Stratified Subsampling**:
- **Target Size**: ~12,000–15,000 frames.
- **100% Retention Guarantee**: Keeps **ALL** instances of rare classes (`bicycle`, `traffic light`, `traffic sign`, `pole`).
- **Sequence Diversity**: Subsamples common vehicle classes uniformly across driving runs.
- **Auto-Format Detection**: Parses Pascal VOC XML and JSON annotation formats.

### To Extract & Convert:
1. Extract your `idd-detection.tar.gz` into `data/raw/idd/`:
   ```bash
   tar -xzf "C:\Users\TUSHAR BASAK\Downloads\idd-detection.tar.gz" -C data/raw/idd/
   ```
2. Run conversion and subsampling:
   ```bash
   python src/utils/dataset_conversion.py --raw-dir data/raw/idd --output-dir data/processed --target-samples 13500
   ```

---

## 🚀 Model Training (`src/train.py`)

The training script automatically inspects available VRAM and configures the optimal batch size and mixed precision (AMP):

| Hardware | Recommended Model | Batch Size | Mixed Precision (AMP) |
| :--- | :--- | :--- | :--- |
| **RTX 2050 (4GB VRAM)** | `yolov8n.pt` | `8` | Enabled |
| **Google Colab (T4 / 16GB)** | `yolov8s.pt` | `16` or `32` | Enabled |

### Launch Training:
```bash
python src/train.py --data configs/data.yaml --epochs 30 --imgsz 640
```
Trained weights will be saved to `models/idd_yolov8/weights/best.pt`.

---

## 🎯 Running the Live Inference Demo (`src/infer.py`)

You can test and demonstrate the prototype immediately — even before custom IDD training completes — by using the `--coco` baseline flag!

### 1. Instant Demo using Webcam:
```bash
python src/infer.py --source 0 --coco --show
```

### 2. Demo on a Video File with Simulated School Hours:
```bash
python src/infer.py --source sample_traffic.mp4 --coco --school-hours --output demo_output.mp4 --show
```

### 3. Demo with Trained IDD Weights:
```bash
python src/infer.py --source sample_traffic.mp4 --weights models/idd_yolov8/weights/best.pt --show
```

---

## 🧪 Module Verification Tests

Both downstream modules are strictly decoupled and can be tested independently:

### Test Geofence & OSM Overpass Engine:
```bash
python src/utils/geofence.py
```
*Validates Point-in-Polygon (PIP) testing against mock Bengaluru school locations.*

### Test Vulnerable Pedestrian Alert Logic:
```bash
python src/modules/pedestrian_alert.py
```
*Validates multi-tier alerts (CRITICAL during school commute hours vs ADVISORY at night).*
