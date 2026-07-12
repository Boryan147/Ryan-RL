import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    # Actor network estimating action probability via logits
    def __init__(self, obs_size, hidden_size, act_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_size)
        )

    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    # Critic network estimating baseline
    def __init__(self, obs_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)