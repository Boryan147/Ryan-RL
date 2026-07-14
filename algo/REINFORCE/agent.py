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

        # empty lists for recording episode
        self.epi_states = []
        self.epi_log_probs = []
        self.epi_rewards = []
    
    def select_action(self, obs):
        # save raw states 
        self.epi_states.append(obs) # state shape (1, obs_size)
        
        logits = self.policy_net(obs)
        dist = Categorical(logits=logits)
        act = dist.sample()

        self.epi_log_probs.append(dist.log_prob(act))
        return act.item()
    
    def store_rew(self, reward):
        self.epi_rewards.append(reward)

    def compute_rewards_to_go(self):
        '''compute total returns iteratively from the end of trajectory'''
        returns = np.zeros_like(self.epi_rewards, dtype=np.float32)
        n = len(self.epi_rewards)
        for i in reversed(range(n)):
            returns[i] = self.epi_rewards[i] + (returns[i+1] * self.config['gamma'] if i+1 < n else 0)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    def optimize(self):
        # calculate total returns
        returns = self.compute_rewards_to_go() # shape: (T,)

        states = torch.cat(self.epi_states).to(self.device) # shape: (T, obs_size)
        log_probs = torch.cat(self.epi_log_probs).to(torch.float32) # shape: (T,)

        # baseline estimation & critic optimization
        values = self.value_net(states).squeeze(-1) # shape: (T,)
        critic_loss = self.critic_criterion(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute advantage( G_t subtracts baseline )
        advantages = returns - values.detach()

        # normalize advantage
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # pseudo-loss for policy gradient
        actor_loss = -(log_probs * advantages).sum()

        # actor optimization
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # reset rollout buffer of episode
        self.epi_states = []
        self.epi_log_probs = []
        self.epi_rewards = [] 
        
