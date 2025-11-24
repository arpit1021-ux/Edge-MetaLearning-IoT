import sys, os
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.optim import Adam

from model.base_cnn import BaseCNN
from model.protonet import proto_loss
from utils.preprocessing import normalize


def train():
    print("Loading dataset...")

    data = np.load("data/features.npy")      # (10299, 128, 6)
    labels = np.load("data/labels.npy")      # (10299,)

    model = BaseCNN()
    opt = Adam(model.parameters(), lr=1e-3)

    unique_classes = np.unique(labels)

    print("Starting meta-learning training...")

    for epoch in range(20):

        # ---- Sample 2 classes per episode ----
        chosen_classes = np.random.choice(unique_classes, size=2, replace=False)

        support_x = []
        support_y = []

        query_x = []
        query_y = []

        for i, c in enumerate(chosen_classes):

            # get indices for this class
            idx = np.where(labels == c)[0]

            # pick 2 samples: 1 support + 1 query
            sample_indices = np.random.choice(idx, size=2, replace=False)

            # support sample
            w1 = data[sample_indices[0]]       # (128,6)
            w1 = normalize(w1)
            w1 = torch.tensor(w1).float().permute(1,0).unsqueeze(0)
            support_x.append(w1)
            support_y.append(i)                # mapped label: 0 or 1

            # query sample
            w2 = data[sample_indices[1]]       # (128,6)
            w2 = normalize(w2)
            w2 = torch.tensor(w2).float().permute(1,0).unsqueeze(0)
            query_x.append(w2)
            query_y.append(i)

        sx = torch.cat(support_x, dim=0)
        sy = torch.tensor(support_y)

        qx = torch.cat(query_x, dim=0)
        qy = torch.tensor(query_y)

        loss = proto_loss(model, sx, sy, qx, qy)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"Epoch: {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model/meta_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
