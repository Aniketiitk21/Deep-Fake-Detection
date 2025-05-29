# DeepFake Image Detection

**Mentor:** Prof. Koteswar Rao Jerripothula, Dept. of Electrical Engineering  
**Duration:** Aug 2024 – Nov 2024

## Overview

Detects deepfake images using deep learning models. Evaluated ResNet50, DenseNet, EfficientNet, and Vision Transformer (ViT) architectures with robust preprocessing and real-time deployment via Flask API.

## Features

- Compared ResNet50, DenseNet, EfficientNet, and ViT for deepfake detection.
- Used MTCNN for face detection and data augmentation for robustness.
- Fine-tuned models with ImageNet weights and optimized hyperparameters.
- Evaluated using Accuracy, F1-score, and ROC-AUC; DenseNet achieved 95% accuracy.
- Deployed best model using Flask API for real-time detection.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch or TensorFlow
- Flask
- OpenCV, MTCNN, scikit-learn

### Installation

```
git clone 
cd deepfake-image-detection
pip install -r requirements.txt
```

### Data Preparation

Organize your dataset:
```
data/
  real/
  fake/
```
MTCNN face detection is applied during preprocessing.

### Training

```
python train.py --model densenet
```
Supported: `resnet50`, `densenet`, `efficientnet`, `vit`

### Evaluation

```
python evaluate.py --model densenet
```

### Inference & API

```
python app.py
```
Send a POST request with an image to `/predict` for real-time detection.

## Results

- DenseNet achieved 95% accuracy.
- Robustness improved with augmentation and MTCNN.
- Real-time detection via Flask API.
``

## License

For academic and research purposes only.
