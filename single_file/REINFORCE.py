import time
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
        return act, probs.log_prob(act), probs.entropy()

    def get_value(self, x):
        return self.valuenet(x)

def reward_to_go(rewards, last_v, gamma):
    returns = np.zeros_like(rewards)
    len_epi = len(rewards)
    for i in reversed(range(len_epi)):
        returns[i] = rewards[i] + gamma * (returns[i+1] if i+1 < len_epi else last_v)
    return returns

def gae(rewards, values, last_v, gamma, gae_lambda):
    deltas = np.zeros_like(rewards)
    n = len(rewards)
    for i in range(n):
        deltas[i] = rewards[i] + gamma * (last_v if i+1 == n else values[i+1]) - values[i]
    gaes = np.zeros_like(rewards)
    for i in reversed(range(n)):
        gaes[i] = deltas[i] + (gamma * gae_lambda * gaes[i+1] if i+1 < n else 0)
    return gaes

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
    run_name = f"CartPole-v1_REINFORCE_seed{seed}__{int(time.time())}"
    writer = SummaryWriter(f'runs/{run_name}')

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger=lambda t: t % 1000 == 0)
    agent = Agent(env.observation_space.shape[0], env.action_space.n).to(device)

    # hyperparameters
    gamma = 0.99
    gae_lambda = 0.97
    lr = 3e-3
    num_epoches = 100
    num_steps = 1000

    optimizer = optim.Adam(agent.policynet.parameters(), lr=5e-3)
    optimizer_value = optim.Adam(agent.valuenet.parameters(), lr=3e-3)
    global_step = 0

    obs, info = env.reset(seed=seed)
    obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0) # shape (1, obs_size)
    done = False
    # training loop
    for epoch in range(num_epoches):
        b_states, b_logprobs, b_vtarget, b_adv = [], [], [], [] 
        states, logprobs, rewards, values = [], [], [], []

        for step in range(num_steps):
            if done:
                obs, info = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            global_step += 1
            act, logprob, _ = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(act.item())
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device).unsqueeze(0)
            done = terminated or truncated
            value = agent.get_value(obs).item() # shape (1, 1)

            states.append(obs)
            rewards.append(reward)
            logprobs.append(logprob)
            values.append(value)

            obs = next_obs

            # log performance metrics
            if 'episode' in info:
                epi_return = info['episode']['r']
                epi_length = info['episode']['l']
                writer.add_scalar('charts/episodic reward', epi_return, global_step)
                writer.add_scalar('charts/episodic length', epi_length, global_step)

            if done or step == num_steps - 1: 
                if terminated:
                    last_v = 0
                elif truncated or step == num_steps - 1:
                    last_v = agent.get_value(obs).item()

                # compute episodic reward-to-go & gae
                v_target = reward_to_go(rewards, last_v, gamma)
                advantages = gae(rewards, values, last_v, gamma, gae_lambda)

                # populate batches
                b_states += states
                b_logprobs += logprobs
                b_vtarget += list(v_target)
                b_adv += list(advantages)

                # empty the episodic lists
                states, logprobs, rewards, values = [], [], [], [] 

        b_states = torch.cat(b_states)  # shape (num_steps, obs_size)
        b_adv = torch.tensor(b_adv, dtype=torch.float32, device=device).detach() # shape (num_steps, )
        b_logprobs = torch.cat(b_logprobs)  # shape (num_steps, )
        b_vtarget = torch.tensor(b_vtarget, dtype=torch.float32, device=device)  # shape (num_steps, )
        _, _, entropy = agent.get_action(b_states)
        entropy_loss = entropy.mean()

        # advantage normalization
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # estimate policy gradient & update policy
        pg_loss = -(b_logprobs * b_adv).mean()
        optimizer.zero_grad()
        pg_loss.backward()
        optimizer.step()

        # define MSE loss for value function & update value
        v_criterion = nn.MSELoss()
        # run multiple times to update value per epoch
        for _ in range(40):
            v_pred = agent.get_value(b_states).view(-1)
            # explained variance of value function
            explained_var = 1 - (b_vtarget - v_pred).var() / (b_vtarget.var() + 1e-8)
            v_loss = v_criterion(v_pred, b_vtarget)
            optimizer_value.zero_grad()
            v_loss.backward()
            optimizer_value.step()

        print(f'Epoch {epoch+1}/{num_epoches}, Global Step: {global_step}')

        # log loss metrics
        writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
        writer.add_scalar('losses/explained_variance', explained_var.item(), global_step)
        writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)

    env.close()
    writer.close()