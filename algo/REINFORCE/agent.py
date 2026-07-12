import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from .model import PolicyNetwork, ValueNetwork

class REINFORCEAgent:
    def __init__(self, obs_size, act_size, config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

        # Instantiate policy network and value network
        self.policy_net = PolicyNetwork(obs_size, config['hidden_size'], act_size).to(self.device)
        self.value_net = ValueNetwork(obs_size, config['baseline_hidden_size']).to(self.device)
        # Define optimizer
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config['lr_actor'])
        self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config['lr_critic'])

        # MSE as loss function for critic
        self.critic_criterion = nn.MSELoss()

        # empty lists for episode
        self.epi_states = []
        self.epi_log_probs = []
        self.epi_rewards = []