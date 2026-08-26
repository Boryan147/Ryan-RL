import argparse
import os 
import random
import numpy as np
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
        logits = self.policynet(x)
        probs = Categorical(logits=logits)
        act = probs.sample()
        return act, probs.log_prob(act)

    def get_value(self, x):
        return self.valuenet(x)

def reward_to_go(rewards, gamma):
    returns = np.zeros_like(rewards)
    len_epi = len(rewards)
    for i in reversed(range(len_epi)):
        returns[i] = rewards[i] + (gamma * rewards[i+1] if i+1 < len_epi else 0)
    return returns

def gae(rewards, values, gamma, gae_lambda):
    deltas = np.zeros_like(rewards)
    n = len(rewards)
    for i in range(n):
        deltas[i] = rewards[i] + gamma * (0 if i+1 == n else values[i+1]) - values[i]
    gaes = np.zeros_like(rewards)
    for i in reversed(range(n)):
        gaes[i] = deltas[i] + (gamma * gae_lambda * deltas[i+1] if i+1 < n else 0)
    return gaes

if __name__ == "__main__":
    # set the seed
    random.seed(15)
    np.random.seed(15)
    torch.manual_seed(15)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(15) 
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, '../videos', episode_trigger=lambda t: t % 100 == 0)
    env.reset(15)
    agent = Agent(env.observation_space.shape[0], env.action_space.n).to(device)

    # hyperparameters
    gamma = 0.99
    gae_lambda = 0.95
    num_epi = 600
    num_traj = 5

    states, logprobs, rewards, values = [], [], [], []
    # training loop
    for i in range(num_epi):
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        done = False

        while not done:
            act, logprob = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(act.item())
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            done = terminated or truncated
            value = agent.get_value(obs)

            states.append(obs)
            rewards.append(reward)
            logprobs.append(logprob)
            values.append(value)

            obs = next_obs
            if done:
                value = agent.get_value(obs)
                values.append(value)

        if i == num_epi:




    env.close()





