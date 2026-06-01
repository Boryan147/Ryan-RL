import torch
import torch.nn as nn

class Dueling_DQN(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions):
        super().__init__()
        # shared structure
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU()
        )
        # Value stream V(s)
        self.Vstream = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Advantage stream A(s,a)
        self.Astream = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        x = self.net(x)
        # aggregation layer
        V = self.Vstream(x)
        A = self.Astream(x)
        return V + A - A.mean(1, keepdim=True)