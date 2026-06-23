# 🚦 YOLO-TS-DRF: Real-Time Traffic Sign Detection with Dynamic Receptive Fields

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![YOLO](https://img.shields.io/badge/YOLO-Traffic%20Sign%20Detection-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Introduction

Traffic Sign Detection (TSD) is a fundamental component of Advanced Driver Assistance Systems (ADAS), Intelligent Transportation Systems (ITS), and autonomous vehicles. Detecting traffic signs accurately is challenging because:

- Traffic signs are often very small in images.
- Illumination and weather conditions vary.
- Occlusions and background clutter exist.
- Scale variation affects feature extraction.

This project proposes **YOLO-TS-DRF**, an enhanced traffic sign detection framework based on YOLO-TS that improves small-object detection through:

1. Dynamic Receptive Field Module (DRF)
2. Small-Object Aware Resolution Optimization (SARO)
3. Adaptive Two-Stage Augmentation Training Strategy (ATATS)

The model is trained on the **TT100K dataset** and achieves **96.0% mAP@50**.

---

# 🎯 Objectives

- Improve traffic sign detection accuracy.
- Enhance small-object detection performance.
- Develop adaptive receptive field learning.
- Preserve fine-grained visual information.
- Maintain real-time inference capability.
- Achieve better performance than baseline YOLO-TS.

---

# 🏗 Proposed Architecture

```text
Input Image (1024×1024)
          │
          ▼
     Backbone
          │
          ▼
  Dynamic Receptive
     Field Module
          │
          ▼
      Neck/FPN
          │
          ▼
 Anchor-Free Head
          │
          ▼
 Traffic Sign Detection
```

---

# 🚀 Novel Contributions

## 1. Dynamic Receptive Field Module (DRF)

### Motivation

Traditional CNNs use fixed receptive fields.

Problem:

- Small signs need local features.
- Large signs need global context.

Fixed kernels cannot adapt efficiently.

### Proposed Solution

The DRF module contains:

- 3×3 Convolution Branch
- 5×5 Convolution Branch
- Dilated 3×3 Branch

Feature fusion:

```
F = w1B1 + w2B2 + w3B3
```

Where:

- B1 = 3×3 branch output
- B2 = 5×5 branch output
- B3 = Dilated branch output
- w1,w2,w3 = learnable attention weights

### Advantages

- Adaptive feature extraction
- Better contextual understanding
- Improved small-object detection

---

## 2. Small-Object Aware Resolution Optimization (SARO)

### Problem

Traffic signs occupy very few pixels.

Example:

```
640×640 image

Stop Sign
20×20 pixels
```

After downsampling:

```
5×5 pixels
```

Important details are lost.

### Solution

Increase input resolution:

```
640×640 → 1024×1024
```

Benefits:

- More pixels per sign
- Better edge information
- Improved localization

---

## 3. Adaptive Two-Stage Augmentation Training Strategy (ATATS)

### Stage 1: Robust Learning

Heavy augmentations:

- Mosaic
- MixUp
- Copy-Paste
- HSV Augmentation

Purpose:

- Learn robust representations
- Prevent overfitting

### Stage 2: Fine-Tuning

Reduced augmentations:

- Mosaic disabled
- Real image training

Purpose:

- Improve localization
- Refine bounding boxes

---

# 📊 Dataset

## TT100K Dataset

Traffic Sign Benchmark from Tsinghua University.

Characteristics:

- High-resolution images
- Multiple traffic sign categories
- Small object challenges
- Real-world driving scenes

Dataset Structure:

```text
TT100K/
│
├── train/
├── val/
├── test/
└── annotations/
```

---

# ⚙️ Training Configuration

| Parameter | Value |
|------------|---------|
| Epochs | 200 |
| Batch Size | 16 |
| Input Size | 1024×1024 |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Momentum | 0.937 |
| Weight Decay | 5e-4 |
| Label Smoothing | 0.1 |
| Scheduler | Cosine Annealing |
| GPU | NVIDIA A100 |

---

# 📈 Results

## Performance Comparison

| Model | mAP@50 | FPS |
|---------|---------|---------|
| YOLO-TS | 92.0% | 137.0 |
| YOLO-TS-DRF | 96.0% | 89.2 |

### Improvement

```
Accuracy Gain = +4.0%
```

### Real-Time Performance

```
89.2 FPS > 30 FPS
```

Suitable for ADAS applications.

---

# 🔬 Ablation Study

| Configuration | mAP@50 |
|---------------|---------|
| YOLO-TS Baseline | 92.0 |
| + DRF | 93.8 |
| + DRF + SARO | 95.0 |
| + DRF + SARO + ATATS | 96.0 |

---

# 📂 Repository Structure

```text
YOLO-TS-DRF/
│
├── datasets/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── models/
│   ├── yolots_drf.yaml
│   └── drf.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   ├── F1_curve.png
│   ├── results.png
│   └── predictions/
│
├── weights/
│   └── best.pt
│
├── train.py
├── detect.py
├── requirements.txt
└── README.md
```

---

# 🛠 Installation

Clone repository:

```bash
git clone https://github.com/yourusername/YOLO-TS-DRF.git
cd YOLO-TS-DRF
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

```bash
python train.py
```

or

```bash
yolo detect train \
model=yolots_drf.yaml \
data=data.yaml \
epochs=200 \
imgsz=1024 \
batch=16
```

---

# 🔍 Inference

```bash
python detect.py
```

or

```bash
yolo detect predict \
model=best.pt \
source=test.jpg
```

---

# 📋 Evaluation Metrics

- Precision
- Recall
- F1 Score
- mAP@50
- FPS

---

# 🔮 Future Work

- Lightweight DRF design
- Edge deployment
- Traffic sign recognition integration
- Adverse weather robustness
- Multi-country traffic sign datasets

---

# 👨‍💻 Author

**Prince Shakya**  
B.Tech, Computer Science and Engineering  
Rajkiya Engineering College, Kannauj

---

# ⭐ Support

If you find this project useful, please consider starring the repository.

```bash
⭐ Star the repository
🍴 Fork the project
📢 Share with others
```
