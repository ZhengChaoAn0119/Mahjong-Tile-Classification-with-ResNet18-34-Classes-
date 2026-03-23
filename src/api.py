from fastapi import FastAPI, File, UploadFile
from src.predict import MahjongClassifier
from pathlib import Path
import io
from PIL import Image

app = FastAPI(title="Mahjong Classifier API")

# 載入模型（啟動就 load）
BASE_DIR = Path(__file__).parent
weight_path = BASE_DIR / "../model/mahjong_resnet18_best.pth"
classifier = MahjongClassifier(str(weight_path))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 讀取上傳圖片
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # 推論
    prediction, confidence = classifier.predict(img)

    return {"prediction": prediction, "confidence": round(confidence, 2)}
