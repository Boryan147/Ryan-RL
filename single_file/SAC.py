import argparse
import time
from datetime import datetime
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal

def make_env(gym_id, seed, idx, run_name, capture_video=True):
    def init_env():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # no ClipAction: tanh squashing keeps actions in bounds, and ClipAction would widen action_space to (-inf, inf)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger=lambda t: t % 50 == 0)
        env.reset(seed=seed)
        return env
    return init_env

# off-policy replay buffer
class Replaybuffer:
    def __init__(self, capacity, obs_shape, act_shape, device):
        self.capacity, self.device = capacity, device
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity,) + act_shape, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminateds = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, next_obs, action, reward, terminated):
        # one transition per env; overwrite the oldest once full
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminateds[self.ptr] = terminated
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return to_t(self.obs), to_t(self.next_obs), to_t(self.actions), to_t(self.rewards), to_t(self.terminateds)

    def __len__(self):
        return self.size

# step 2: soft Q-network Q(s, a). Off-policy data needs Q, not V: the target samples a' from the
# *current* policy, so transitions collected by old policies stay valid
class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))

LOG_STD_MIN, LOG_STD_MAX = -5, 2

# step 4: tanh-squashed Gaussian policy with state-dependent std
class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        # map tanh output [-1, 1] to the env's action bounds
        high, low = envs.single_action_space.high, envs.single_action_space.low
        self.register_buffer('action_scale', torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def get_action(self, x):
        h = self.net(x)
        mean = self.fc_mean(h)
        # squash log_std smoothly into [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.tanh(self.fc_logstd(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        probs = Normal(mean, log_std.exp())
        u = probs.rsample()  # reparameterization trick: u = mean + std * eps, so gradients flow into mean/std
        y = torch.tanh(u)
        action = y * self.action_scale + self.action_bias
        # change of variables: log pi(a) = log N(u) - sum log |da/du|,  da/du = scale * (1 - tanh(u)^2)
        log_prob = probs.log_prob(u) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias  # deterministic action for eval
        return action, log_prob, mean_action

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym-id', type=str, default='HalfCheetah-v5')
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--total-timesteps', type=int, default=300000)
    parser.add_argument('--capture-video', type=int, default=1)
    args = parser.parse_args()

    # common arguments
    gym_id = args.gym_id
    total_timesteps = args.total_timesteps
    num_envs = 1
    buffer_size = int(1e6)
    batch_size = 256
    learning_starts = 5000  # uniform random actions until the buffer has this many steps

    # hyperparameters (CleanRL sac_continuous_action defaults)
    gamma = 0.99
    tau = 0.005             # polyak averaging rate for target Q-networks
    policy_lr = 3e-4
    q_lr = 1e-3
    policy_frequency = 2    # delayed actor update (TD3 trick)
    target_network_frequency = 1
    autotune = True         # learn alpha instead of fixing it
    alpha = 0.2             # only used when autotune is False

    # set the seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True

    device = torch.device('cpu')
    run_name = f"{gym_id}_SAC_seed{seed}__{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    writer = SummaryWriter(f'runs/{run_name}')

    # SAME_STEP: a finished env resets inside the same step(); its real last obs is in infos['final_obs']
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id, seed + i, i, run_name, capture_video=bool(args.capture_video)) for i in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), 'only continuous action space is allowed'
    envs.action_space.seed(seed)
    print('envs.single_observation_space.shape', envs.single_observation_space.shape)
    print('envs.single_action_space.shape', envs.single_action_space.shape)
    actor = Actor(envs).to(device)
    # twin Q-networks (min of two counters overestimation) + slowly-tracking target copies
    qf1, qf2 = QNetwork(envs).to(device), QNetwork(envs).to(device)
    qf1_target, qf2_target = QNetwork(envs).to(device), QNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=policy_lr)

    # step 6: automatic entropy tuning, keep policy entropy near target_entropy = -|A|
    if autotune:
        target_entropy = -float(np.prod(envs.single_action_space.shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=q_lr)

    rb = Replaybuffer(buffer_size, envs.single_observation_space.shape, envs.single_action_space.shape, device)

    obs, infos = envs.reset(seed=seed)
    start_time = time.time()
    # training loop: one env step, then one gradient step
    for global_step in range(total_timesteps):
        if global_step < learning_starts:
            acts = envs.action_space.sample()
        else:
            with torch.no_grad():
                acts, _, _ = actor.get_action(torch.tensor(obs, dtype=torch.float32, device=device))
            acts = acts.cpu().numpy()

        next_obs, rewards, terminateds, truncateds, infos = envs.step(acts)

        # next_obs of a finished env is already the reset obs of its new episode -> swap in the real last obs
        real_next_obs = next_obs.copy()
        if 'final_obs' in infos:
            for idx in range(num_envs):
                if infos['_final_obs'][idx]:
                    real_next_obs[idx] = infos['final_obs'][idx]
        # store terminated, not done: a truncated episode must still bootstrap from s'
        for idx in range(num_envs):
            rb.push(obs[idx], real_next_obs[idx], acts[idx], rewards[idx], terminateds[idx])

        obs = next_obs

        # log performance metrics
        # with SAME_STEP, episode stats of finished envs live in infos['final_info']
        if 'final_info' in infos:
            for idx in range(num_envs):
                if infos['_final_info'][idx]:
                    writer.add_scalar('charts/episodic reward', infos['final_info']['episode']['r'][idx], global_step)
                    writer.add_scalar('charts/episodic length', infos['final_info']['episode']['l'][idx], global_step)
                    print(f"global_step={global_step}, episodic reward={infos['final_info']['episode']['r'][idx]:.1f}")

        if global_step >= learning_starts:
            b_obs, b_next_obs, b_act, b_rew, b_term = rb.sample(batch_size)

            # step 3: soft Bellman target  y = r + gamma * (1 - terminated) * (min Q_targ(s', a') - alpha * log pi(a'|s')),  a' ~ pi(.|s')
            with torch.no_grad():
                next_act, next_logp, _ = actor.get_action(b_next_obs)
                qf1_next = qf1_target(b_next_obs, next_act)
                qf2_next = qf2_target(b_next_obs, next_act)
                min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_logp
                target_q = b_rew + gamma * (1.0 - b_term) * min_qf_next.flatten()

            qf1_a = qf1(b_obs, b_act).flatten()
            qf2_a = qf2(b_obs, b_act).flatten()
            qf1_loss = ((qf1_a - target_q) ** 2).mean()
            qf2_loss = ((qf2_a - target_q) ** 2).mean()
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # step 5: actor loss  E_s[ alpha * log pi(a|s) - min Q(s, a) ],  a ~ pi(.|s) via rsample
            if global_step % policy_frequency == 0:
                # do policy_frequency actor steps to keep the actor/critic update ratio at 1:1
                for _ in range(policy_frequency):
                    pi, logp, _ = actor.get_action(b_obs)
                    min_qf_pi = torch.min(qf1(b_obs, pi), qf2(b_obs, pi))
                    actor_loss = (alpha * logp - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if autotune:
                        # alpha rises when entropy (-log pi) falls below target, and falls when above
                        with torch.no_grad():
                            _, logp, _ = actor.get_action(b_obs)
                        alpha_loss = (-log_alpha.exp() * (logp + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # step 3: target networks slowly track the online ones (polyak averaging)
            if global_step % target_network_frequency == 0:
                with torch.no_grad():
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.mul_(1 - tau).add_(tau * param)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.mul_(1 - tau).add_(tau * param)

            # log loss metrics
            if global_step % 1000 == 0:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar('losses/qf1_values', qf1_a.mean().item(), global_step)
                writer.add_scalar('losses/qf2_values', qf2_a.mean().item(), global_step)
                writer.add_scalar('losses/qf1_loss', qf1_loss.item(), global_step)
                writer.add_scalar('losses/qf2_loss', qf2_loss.item(), global_step)
                writer.add_scalar('losses/qf_loss', qf_loss.item() / 2.0, global_step)
                writer.add_scalar('losses/actor_loss', actor_loss.item(), global_step)
                writer.add_scalar('losses/alpha', alpha, global_step)
                writer.add_scalar('losses/entropy', -logp.mean().item(), global_step)
                if autotune:
                    writer.add_scalar('losses/alpha_loss', alpha_loss.item(), global_step)
                writer.add_scalar('charts/SPS', sps, global_step)
                print(f'global_step={global_step}, SPS={sps}, alpha={alpha:.3f}, qf1={qf1_a.mean().item():.1f}')

    envs.close()
    writer.close()
