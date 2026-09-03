import time
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(gym_id, seed, idx, run_name, capture_video=True):
    def init_env():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger=lambda t: t % 1000 == 0)
        env.reset(seed=seed)
        return env
    return init_env

# Policy and value networks
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actornet = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        )

        self.valuenet = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(envs.single_action_space.shape[0]))

    def get_action_and_value(self, x, act=None):
        # mean = self.actornet(x)
        # std = self.actor_logstd.exp()
        # probs = Normal(loc=mean, scale=std)
        logits = self.actornet(x)
        probs = Categorical(logits=logits)
        if act is None:
            act = probs.sample()
        return act, probs.log_prob(act), probs.entropy(), self.valuenet(x)
        # return act, probs.log_prob(act).sum(dim=-1), probs.entropy().sum(dim=-1), self.valuenet(x)

    def get_value(self, x):
        return self.valuenet(x)

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
    # common arguments
    gym_id = 'CartPole-v1'
    num_epoches = 50
    num_steps = 128
    num_envs = 4
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
    run_name = f"{gym_id}_A2C_seed{seed}__{int(time.time())}"
    writer = SummaryWriter(f'runs/{run_name}')

    envs = gym.vector.SyncVectorEnv([make_env(gym_id, seed + i, i, run_name) for i in range(num_envs)])
    agent = Agent(envs).to(device)

    # hyperparameters
    gamma = 0.99
    gae_lambda = 0.97
    po_lr = 3e-4
    v_lr = 1e-3
    ent_coef = 0.01

    # actor_params = list(agent.actornet.parameters()) + [agent.actor_logstd]
    optimizer_actor = optim.Adam(agent.actornet.parameters(), lr=po_lr)
    optimizer_value = optim.Adam(agent.valuenet.parameters(), lr=v_lr)

    obs, infos = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    done = False
    global_step = 0

    # training loop
    for epoch in range(num_epoches):
        # storage buffer setup
        obs_buf = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape, device=device)
        act_buf = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape, device=device)
        rew_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        val_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        adv_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        ret_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        start_idx = 0

        with torch.no_grad():
            for step in range(num_steps):
                if done:
                    obs, info = envs.reset()
                    obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                global_step += 1
                act, logprob, _, value = agent.get_action_and_value(obs)
                act_np = np.clip(act.cpu().numpy().flatten(), env.action_space.low, env.action_space.high)
                next_obs, reward, terminated, truncated, info = env.step(act_np)
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device).unsqueeze(0)
                done = terminated or truncated

                obs_buf[step] = obs
                act_buf[step] = act
                rew_buf[step] = reward
                val_buf[step] = value.item()

                obs = next_obs
                if done or step == num_steps - 1: 
                    last_v = 0.0 if terminated else agent.get_value(obs).item()
                        
                    slice_ = slice(start_idx, step + 1)
                    # compute gae & TD target and populate buffer
                    adv_buf[slice_] = gae(rew_buf[slice_], val_buf[slice_], last_v, gamma, gae_lambda)
                    ret_buf[slice_] = adv_buf[slice_] + val_buf[slice_]

                    start_idx = step + 1

                # log performance metrics
                if 'episode' in info:
                    epi_return = info['episode']['r']
                    epi_length = info['episode']['l']
                    writer.add_scalar('charts/episodic reward', epi_return, global_step)
                    writer.add_scalar('charts/episodic length', epi_length, global_step)

        _, logp_buf, entropy, _ = agent.get_action_and_value(obs_buf, act=act_buf)
        entropy_loss = entropy.mean()

        # advantage normalization
        b_adv = torch.from_numpy(adv_buf).to(device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # estimate policy gradient & update policy
        pg_loss = -(logp_buf * b_adv).mean() - ent_coef * entropy_loss
        optimizer_actor.zero_grad()
        pg_loss.backward()
        nn.utils.clip_grad_norm_(agent.actornet.parameters(), max_norm=0.5)
        optimizer_actor.step()

        # define MSE loss for value function & update value
        b_ret = torch.from_numpy(ret_buf).to(device)
        # run multiple times to update value per epoch
        for _ in range(40):
            v_pred = agent.get_value(obs_buf).view(-1)
            # explained variance of value function
            explained_var = 1 - (b_ret - v_pred).var() / (b_ret.var() + 1e-8)
            v_loss = 0.5 * ((b_ret - v_pred) ** 2).mean()
            optimizer_value.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(agent.valuenet.parameters(), max_norm=0.5)
            optimizer_value.step()

        print(f'Epoch {epoch+1}/{num_epoches}, Global Step: {global_step}')

        # log loss metrics
        writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
        writer.add_scalar('losses/explained_variance', explained_var.item(), global_step)
        writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)

    envs.close()
    writer.close()