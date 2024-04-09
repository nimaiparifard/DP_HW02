# Define the MLP architecture
import torch
from torch import nn, optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):

    def __init__(self, activation_fn=nn.ReLU):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(27, 64),
            activation_fn(),
            nn.Linear(64, 128),
            activation_fn(),
            nn.Linear(128, 256),
            activation_fn(),
            nn.Linear(256, 128),
            activation_fn(),
            nn.Linear(128, 64),
            activation_fn(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.layers(x)