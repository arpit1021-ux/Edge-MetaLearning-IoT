from fastapi import FastAPI
import numpy as np
import torch
from model.base_cnn import BaseCNN

app = FastAPI()

# Load model
model = BaseCNN()
model.load_state_dict(torch.load("model/meta_model.pth", map_location="cpu"))
model.eval()

@app.post("/predict")
def predict(data: dict):
    window = np.array(data["window"]).reshape(1,6,128)
    x = torch.tensor(window).float()
    emb = model(x)
    return {"embedding": emb.detach().numpy().tolist()}
