import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================
# 參數設定
# ======================
DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.0003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Data Transform
# ======================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.join(
    DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(
    DATA_DIR, "val"), transform=val_transform)

# ======================
# 類別分布檢查
# ======================

labels = [label for _, label in train_dataset]
print("Class distribution:", Counter(labels))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

num_classes = len(train_dataset.classes)

# ======================
# Model: ResNet18
# ======================
model = models.resnet18(weights="IMAGENET1K_V1")

# 1️⃣ 載入模型
model = models.resnet18(weights="IMAGENET1K_V1")

# 2️⃣ 替換分類層（先做）
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 3️⃣ 凍結全部
for param in model.parameters():
    param.requires_grad = False

# 4️⃣ 打開 layer4
for param in model.layer4.parameters():
    param.requires_grad = True

# 5️⃣ 打開 fc（保險寫法）
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# ======================
# Loss / Optimizer
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {"params": model.fc.parameters(), "lr": 1e-4},
    {"params": model.layer4.parameters(), "lr": 1e-5},
])

# ======================
# 訓練
# ======================
best_acc = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ===== Train =====
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # ===== Validation =====
    model.eval()
    val_correct = 0
    val_total = 0
    wrong_images = []
    wrong_labels = []
    wrong_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            # 找出錯誤的圖片
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    wrong_images.append(images[i].cpu())
                    wrong_labels.append(labels[i].cpu().item())
                    wrong_preds.append(predicted[i].cpu().item())

    val_acc = val_correct / val_total

    print("Classes:", train_dataset.classes)
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc:   {val_acc:.4f}")

    # 儲存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "mahjong_resnet18_best.pth")
        print("✅ Model Saved")

print("\nTraining Finished")
print("Best Val Acc:", best_acc)

# helper for plotting


def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    plt.imshow(img)


# 顯示前 16 張錯誤圖片
num_show = min(16, len(wrong_images))
plt.figure(figsize=(12, 12))
for i in range(num_show):
    plt.subplot(4, 4, i+1)
    imshow(wrong_images[i])
    plt.title(
        f"T:{train_dataset.classes[wrong_labels[i]]}\nP:{train_dataset.classes[wrong_preds[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
