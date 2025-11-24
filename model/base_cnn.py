import torch
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6,32,5,padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64,64)
        )
    
    def forward(self,x):
        return self.net(x)
