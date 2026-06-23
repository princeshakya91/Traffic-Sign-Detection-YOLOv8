YOLO-TS-DRF: Real-Time Traffic Sign Detection Using Dynamic Receptive Fields
📌 Overview

This project presents YOLO-TS-DRF, an enhanced traffic sign detection framework designed to improve the detection of small traffic signs in real-world driving environments. The proposed model is built upon the YOLO-TS architecture and introduces three key enhancements:

Dynamic Receptive Field (DRF) Module
Small-Object Aware Resolution Optimization (SARO)
Adaptive Two-Stage Augmentation Training Strategy (ATATS)

The model is trained and evaluated on the TT100K dataset and achieves 96.0% mAP@50, outperforming the baseline YOLO-TS model while maintaining real-time performance.

🎯 Project Objectives
Improve traffic sign detection accuracy for small objects.
Enhance feature extraction through adaptive receptive fields.
Preserve fine-grained traffic sign details using higher input resolution.
Improve model generalization using stage-wise augmentation training.
Maintain real-time inference capability for Intelligent Transportation Systems (ITS) and ADAS applications.
🏗️ Proposed Architecture

The proposed YOLO-TS-DRF framework consists of:

1. Backbone Network

Extracts hierarchical visual features from input images.

2. Dynamic Receptive Field (DRF) Module

Introduces adaptive multi-scale feature extraction through:

3×3 Convolution Branch
5×5 Convolution Branch
Dilated 3×3 Convolution Branch
Attention-Based Feature Fusion
3. Neck Network

Performs multi-scale feature fusion to enhance object representation.

4. Anchor-Free Detection Head

Predicts:

Bounding Boxes
Object Confidence
Traffic Sign Classes
🚀 Proposed Novelties
1️⃣ Dynamic Receptive Field (DRF)

Traditional convolution layers use fixed receptive fields, which are not optimal for objects of varying sizes.

DRF dynamically combines:

Local Features (3×3)
Medium Context Features (5×5)
Global Context Features (Dilated Convolution)

This allows the network to adaptively focus on traffic signs of different scales.

2️⃣ Small-Object Aware Resolution Optimization (SARO)

Traffic signs in TT100K are often extremely small.

SARO increases the input resolution from:

640 × 640 → 1024 × 1024

Benefits:

Preserves fine details
Improves visibility of distant traffic signs
Enhances small-object detection performance
3️⃣ Adaptive Two-Stage Augmentation Training Strategy (ATATS)

Training is divided into two stages:

Stage 1: Robust Feature Learning

Heavy augmentations:

Mosaic
MixUp
Copy-Paste
Color Jitter
Stage 2: Fine-Tuning

Reduced augmentation:

Mosaic disabled
Real image training
Improved localization accuracy
📊 Dataset
TT100K Dataset

The TT100K dataset contains:

High-resolution traffic scene images
Multiple traffic sign categories
Significant scale variations
Small object detection challenges

Official Website:

https://cg.cs.tsinghua.edu.cn/traffic-sign/

⚙️ Training Configuration
Parameter	Value
Epochs	200
Batch Size	16
Input Resolution	1024 × 1024
Optimizer	SGD
Learning Rate	0.01
Weight Decay	5e-4
Label Smoothing	0.1
Scheduler	Cosine Annealing
GPU	NVIDIA A100
📈 Results
Model	mAP@50	FPS
YOLO-TS (Baseline)	92.0%	137.0
YOLO-TS-DRF (Proposed)	96.0%	89.2
Key Improvements

✅ +4.0% mAP@50 improvement

✅ Better small-object detection

✅ Adaptive feature extraction

✅ Real-time inference maintained (>30 FPS)

📂 Project Structure
YOLO-TS-DRF/
│
├── datasets/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── models/
│   ├── YOLO_TS_DRF.yaml
│   └── drf.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   ├── F1_curve.png
│   └── sample_predictions/
│
├── train.py
├── detect.py
├── requirements.txt
├── README.md
└── best.pt
🛠️ Technologies Used
Python
PyTorch
Ultralytics YOLO
OpenCV
NumPy
Matplotlib
Pandas
📋 Evaluation Metrics

The model is evaluated using:

Precision
Recall
F1-Score
mAP@50
Inference FPS
🔮 Future Work
Lightweight DRF design for edge devices
Deployment on embedded systems
Traffic sign recognition integration
Adverse weather adaptation
Multi-dataset evaluation

👨‍💻 Author

Prince Shakya
B.Tech, Computer Science and Engineering
Rajkiya Engineering College, Kannauj
