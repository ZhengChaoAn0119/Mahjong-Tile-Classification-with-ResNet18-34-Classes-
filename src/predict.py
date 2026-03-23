# predict.py
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision import datasets
from src.model import get_model
import os


def get_class_names(dataset_path):
    class_names = sorted(os.listdir(dataset_path))
    return class_names


class MahjongClassifier:
    def __init__(self, weight_path,):
        self.model = get_model(num_classes=34)
        self.model.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset_path = os.path.join(os.path.dirname(
            __file__), "..", "dataset", "train")
        self.class_names = get_class_names(dataset_path)

    def predict(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img)
            probs = F.softmax(output, dim=1)

            confidence, pred = torch.max(probs, 1)

        return self.class_names[pred.item()], confidence.item()
