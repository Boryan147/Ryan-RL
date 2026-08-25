import argparse
import os 
import random
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Policy and value networks
class Agent(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.policynet = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, act_size)
        )

        self.valuenet = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_action(self, x):
        return self.policynet(x)

    def get_value(self, x):
        return self.valuenet(x)

if __name__ == "__main__":
    # set seed

    env = gym.make('CartPole-v1')
    env = gym.wrappers.RecordEpisodeStatistics(env)

    agent = Agent(env.observation_space.shape[0], env.action_space.n)


