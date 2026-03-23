# model.py
import torch.nn as nn
from torchvision import models


def get_model(num_classes=34):
    """
    建立 ResNet18 並修改最後一層 fc 以符合你的分類數
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
