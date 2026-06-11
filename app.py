import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transfer_learning.predict import inference
from transfer_learning.restnet18 import ResNetFinetuner

app = FastAPI()
templates = Jinja2Templates(directory="templates")
device = "cuda" if torch.cuda.is_available() else "cpu"
label2id = {'cat': 0, 'dog': 1}
id2label = {v: k for k, v in label2id.items()}

model = ResNetFinetuner.load_from_checkpoint(
    "model/best-model-transfer-learning.ckpt",
    id2label=id2label,
    label2id=label2id
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label, confidence = inference(model=model, image=image,
                                  device=device, id2label=id2label)

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }