# Define the MLP architecture
import torch
from torch import nn, optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.layers(x)