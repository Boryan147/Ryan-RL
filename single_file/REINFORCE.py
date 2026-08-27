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
        returns[i] = rewards[i] + (gamma * returns[i+1] if i+1 < len_epi else 0)
    return returns

def gae(rewards, values, gamma, gae_lambda):
    deltas = np.zeros_like(rewards)
    n = len(rewards)
    for i in range(n):
        deltas[i] = rewards[i] + gamma * (0 if i+1 == n else values[i+1]) - values[i]
    gaes = np.zeros_like(rewards)
    for i in reversed(range(n)):
        gaes[i] = deltas[i] + (gamma * gae_lambda * gaes[i+1] if i+1 < n else 0)
    return torch.tensor(gaes, dtype=torch.float32).to(device).detach()

if __name__ == "__main__":
    # set the seed
    seed = 15
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, '../videos', episode_trigger=lambda t: t % 100 == 0)
    env.reset(seed=seed)
    agent = Agent(env.observation_space.shape[0], env.action_space.n).to(device)

    # hyperparameters
    gamma = 0.99
    gae_lambda = 0.95
    lr = 3e-3
    num_epi = 600

    optimizer = optim.Adam(agent.policynet.parameters(), lr=lr)
    optimizer_value = optim.SGD(agent.valuenet.parameters(), lr=lr)
   
    # training loop
    for i in range(num_epi):
        logprobs, rewards, values = [], [], []
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0) # shape (1, obs_size)
        done = False

        while not done:
            act, logprob = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(act.item())
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device).unsqueeze(0)
            done = terminated or truncated
            value = agent.get_value(obs) # shape (1, 1)

            rewards.append(reward)
            logprobs.append(logprob)
            values.append(value)

            obs = next_obs
        value = agent.get_value(obs)
        values.append(value)

        # estimate policy gradient & update policy
        pg_loss = -(torch.cat(logprobs) * gae(rewards, values, gamma, gae_lambda)).sum()
        optimizer.zero_grad()
        pg_loss.backward()
        optimizer.step()

        # define MSE loss for value function & update value
        v_criterion = nn.MSELoss()
        v_loss = v_criterion(
            torch.cat(values[:-1]).view(-1),  
            torch.tensor(reward_to_go(rewards, gamma), dtype=torch.float32).to(device)
            )
        optimizer_value.zero_grad()
        v_loss.backward()
        optimizer_value.step()

        if i % 50 == 0:
            print(f"Episode {i}, Return: {sum(rewards)}")

    env.close()





