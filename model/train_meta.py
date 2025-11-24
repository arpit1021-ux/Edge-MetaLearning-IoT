import torch
import numpy as np
from torch.optim import Adam
from model.base_cnn import BaseCNN
from model.protonet import proto_loss
from utils.preprocessing import create_windows, normalize

def train():
    print("Loading dataset...")
    
    # You MUST create these files after preprocessing your data!
    data = np.load("data/features.npy")     # shape: (N, 128, 6)
    labels = np.load("data/labels.npy")     # shape: (N,)
    
    print("Starting meta-learning training...")
    model = BaseCNN()
    opt = Adam(model.parameters(), lr=1e-3)

    # Simple meta-learning loop (demo)
    for epoch in range(20):
        idx = np.random.randint(0, len(data)-128)
        
        window = data[idx:idx+128]
        window = normalize(window)
        
        # reshape to (1,6,128)
        window = torch.tensor(window).float().permute(1,0).unsqueeze(0)

        sx = window
        sy = torch.tensor([labels[idx]])

        qx = window
        qy = torch.tensor([labels[idx]])

        loss = proto_loss(model, sx, sy, qx, qy)
        opt.zero_grad()
        loss.backward()
        opt.step()

        print("Epoch:", epoch, "| Loss:", loss.item())

    torch.save(model.state_dict(), "model/meta_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
