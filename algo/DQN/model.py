import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, n_observatoins, hidden_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observatoins, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    def forward(self, x):
        return self.net(x)
