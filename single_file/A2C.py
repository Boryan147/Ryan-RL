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
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger=lambda t: t % 300 == 0)
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
            layer_init(nn.Linear(64, envs.single_action_space.shape[0]), std=0.01)
        )

        self.valuenet = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(envs.single_action_space.shape[0]))

    def get_action_and_value(self, x, act=None):
        mean = self.actornet(x)
        std = self.actor_logstd.exp()
        probs = Normal(loc=mean, scale=std)
        # logits = self.actornet(x)
        # probs = Categorical(logits=logits)
        if act is None:
            act = probs.sample()
        # return act, probs.log_prob(act), probs.entropy(), self.valuenet(x)
        return act, probs.log_prob(act).sum(dim=-1), probs.entropy().sum(dim=-1), self.valuenet(x)

    def get_value(self, x):
        return self.valuenet(x)

def gae(rewards, values, dones, last_v, gamma, gae_lambda):
    deltas = torch.zeros_like(rewards).to(device)
    n = len(rewards)
    for i in range(n):
        deltas[i] = rewards[i] + gamma * (last_v if i+1 == n else values[i+1]) * (1.0 - dones[i]) - values[i]
    gaes = torch.zeros_like(rewards).to(device)
    for i in reversed(range(n)):
        gaes[i] = deltas[i] + (gamma * gae_lambda * gaes[i+1] * (1.0 - dones[i]) if i+1 < n else 0)
    return gaes

if __name__ == "__main__":
    # common arguments
    gym_id = 'Pendulum-v1'
    num_epoches = 2000
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
    assert isinstance(envs.single_action_space, gym.spaces.Box), 'only continuous action space is allowed'
    print('envs.single_observation_space.shape', envs.single_observation_space.shape)
    print('envs.single_action_space.shape', envs.single_action_space.shape)
    agent = Agent(envs).to(device)

    # hyperparameters
    gamma = 0.99
    gae_lambda = 0.97
    lr = 1e-3
    ent_coef = 0.0
    vf_coef = 0.5

    actor_params = list(agent.actornet.parameters()) + [agent.actor_logstd]
    optimizer_actor = optim.Adam(actor_params, lr=3e-4)
    optimizer_critic = optim.Adam(agent.valuenet.parameters(), lr=1e-3)

    obs, infos = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device) # shape (num_envs, obs_size)
    global_step = 0
    # training loop
    for epoch in range(num_epoches):
        # storage buffer setup
        obs_buf = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape, device=device)
        act_buf = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape, device=device)
        rew_buf = torch.zeros((num_steps, num_envs), device=device)
        val_buf = torch.zeros((num_steps, num_envs), device=device)
        done_buf = torch.zeros((num_steps, num_envs), device=device)

        # rollout data collection
        with torch.no_grad():
            for step in range(num_steps):
                global_step += 1 * num_envs
                acts, logprob, _, value = agent.get_action_and_value(obs)
                act_np = np.clip(acts.cpu().numpy(), envs.single_action_space.low, envs.single_action_space.high)
                next_obs, rewards, terminateds, truncateds, infos = envs.step(act_np)
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                done = np.logical_or(terminateds, truncateds)
                done = torch.tensor(done).to(device)

                obs_buf[step] = obs
                act_buf[step] = acts
                rew_buf[step] = torch.tensor(rewards, dtype=torch.float32).to(device)
                val_buf[step] = value.flatten() 
                done_buf[step] = done

                obs = next_obs

                # log performance metrics
                if 'episode' in infos:
                    for idx in range(num_envs):
                        if infos['_episode'][idx]:
                            writer.add_scalar('charts/episodic reward', infos['episode']['r'][idx], global_step) 
                            writer.add_scalar('charts/episodic length', infos['episode']['l'][idx], global_step)

        with torch.no_grad():
            last_v = agent.valuenet(obs).flatten()
            adv_buf = gae(rew_buf, val_buf, done_buf, last_v, gamma, gae_lambda)
            ret_buf = adv_buf + val_buf

        _, logp_buf, entropy, v_pred = agent.get_action_and_value(obs_buf, act=act_buf)
        entropy_loss = entropy.mean()

        # flatten the buffer
        b_logprob = logp_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_ret = ret_buf.reshape(-1)
        
        # advantage normalization
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # estimate policy gradient & update policy
        pg_loss = -(b_logprob * b_adv).mean()

        # define MSE loss for value function & update value
        v_pred = v_pred.reshape(-1)
        # explained variance of value function
        explained_var = 1 - (b_ret - v_pred).var() / (b_ret.var() + 1e-8)
        v_loss = 0.5 * ((b_ret - v_pred) ** 2).mean()

        # total_loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
        optimizer_actor.zero_grad()
        pg_loss.backward()
        nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(agent.valuenet.parameters(), max_norm=1.0)
        optimizer_critic.step()

        print(f'Epoch {epoch+1}/{num_epoches}, Global Step: {global_step}')

        # log loss metrics
        writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
        writer.add_scalar('losses/explained_variance', explained_var.item(), global_step)
        writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)

    envs.close()
    writer.close()