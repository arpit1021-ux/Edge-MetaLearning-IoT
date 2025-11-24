import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

from model.base_cnn import BaseCNN
from utils.preprocessing import normalize

app = FastAPI()

# Load meta-learned model
MODEL_PATH = "model/meta_model.pth"
model = BaseCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


class SensorWindow(BaseModel):
    window: list  # 128×6 sensor window


def preprocess(window):
    """Convert 128×6 sensor window → torch (1,6,128)"""
    w = np.array(window)
    w = normalize(w)
    w = torch.tensor(w).float().permute(1, 0).unsqueeze(0)
    return w


@app.post("/predict")
def predict(data: SensorWindow):
    x = preprocess(data.window)
    with torch.no_grad():
        embedding = model(x)           # (1,64)
        score = torch.norm(embedding)  # just a placeholder score

    return {
        "prediction": float(score.item())
    }


@app.get("/")
def home():
    return {"message": "Edge IoT Meta-Learning API is running"}
