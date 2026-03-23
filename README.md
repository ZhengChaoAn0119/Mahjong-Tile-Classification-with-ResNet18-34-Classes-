# Mahjong Tile Classification with ResNet18 (34 Classes)

## 📌 Overview

This project implements a **Mahjong tile classification system** using a fine-tuned ResNet18 model.
It supports **34 classes** (man, pin, sou, and honor tiles) and provides both:

* 🔹 Model inference via Python
* 🔹 REST API using FastAPI

The trained model achieves approximately **97% accuracy** on the validation set.

---

## 🧠 Features

* ResNet18-based image classification (PyTorch)
* 34 Mahjong tile categories
* Clean modular design (`model / predict / api`)
* FastAPI deployment for inference
* Ready-to-run demo images
* Reproducible training pipeline

---

## 🗂️ Project Structure

```
majsoul_project/
├── dataset/                # Training & validation dataset (optional / partial)
│   ├── train/
│   └── val/
├── demo_images/            # Demo images for testing
├── model/
│   └── mahjong_resnet18_best.pth
├── requirements.txt
├── README.md
└── src/
    ├── model.py           # Model definition (ResNet18)
    ├── predict.py         # Inference wrapper (MahjongClassifier)
    ├── train.py           # Training pipeline
    └── api.py             # FastAPI service
```

---

## 🚀 Installation

```bash
git clone <your-repo-url>
cd majsoul_project

python -m venv .venv
.venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## 🧪 Run Inference (Python)

```python
from src.predict import MahjongClassifier

classifier = MahjongClassifier("model/mahjong_resnet18_best.pth")
pred, conf = classifier.predict("demo_images/demo001.png")

print(pred, conf)
```

---

## 🌐 Run API (FastAPI)

```bash
uvicorn src.api:app --reload
```

Open browser:

```
http://127.0.0.1:8000/docs
```

Upload an image and get:

```json
{
  "prediction": "3m",
  "confidence": 0.97
}
```

---

## 📦 API Example (Python Request)

```python
import requests

url = "http://127.0.0.1:8000/predict"

with open("demo_images/demo001.png", "rb") as f:
    files = {"file": f}
    res = requests.post(url, files=files)

print(res.json())
```

---

## 🏋️ Training

```bash
python -m src.train
```

* Uses `dataset/train` and `dataset/val`
* Model: ResNet18
* Loss: CrossEntropyLoss
* Optimizer: Adam

Output model will be saved to:

```
model/mahjong_resnet18_trained.pth
```

---

## 📊 Dataset

Dataset is organized using PyTorch `ImageFolder` format:

```
dataset/
├── train/
│   ├── 1m/
│   ├── 1p/
│   └── ...
└── val/
```

Each folder represents a Mahjong tile class.

> Note: Only a subset of the dataset may be included due to size limitations.

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Torchvision
* FastAPI
* Uvicorn
* Pillow

---