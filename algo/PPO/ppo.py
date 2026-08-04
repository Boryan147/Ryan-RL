import argparse
import os 
from distutils.util import strtobool
import random
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def parse_args():
    '''set up some common arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip('.py'), 
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default='CartPole-v1', help='gym environment')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=15, help='random seed of the experiment')
    parser.add_argument('--timesteps', type=int, default=25000, help='total timesteps of the experiment')
    # help reproduce experiment
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, torch.backends.cudnn.deterministic=False')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled be default')
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out videos folder)")
    
    # Algorithm related arguments
    parser.add_argument('--num-envs', type=int, default=4, help='the number of environments in parallel')
    parser.add_argument('--num-steps', type=int, default=128, help='the number of steps the agent takes for rollout data')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='GAE for advantage estimation')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='lambda for GAE')
    parser.add_argument('--num-minibatches', type=int, default=4, help='the number of mini-batches')
    parser.add_argument('--epoches', type=int, default=4, help='the K epoches to update the policy')
    args = parser.parse_args()
    args.batch_size = int(args.num_steps * args.num_envs)
    args.mini_batch_size = int(args.batch_size // args.num_minibatches)
    return args

def make_env(gym_id, seed, idx, capture_video, run_name):
        def init_env():
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                 if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger=lambda t: t % 1000 == 0) 
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return init_env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        )
        self.critic = nn.Sequential(
            layer_init(nn.linear(np.array(envs.single_observation_space.shape).prod()), 64),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None): # action ???
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":
    args = parse_args()
    run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}'
    writer = SummaryWriter(f'runs/{run_name}')

    # set the seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) 
            for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), 'only discrete action space is allowed'
    print('envs.single_observation_space.shape', envs.single_observation_space.shape)
    print('envs.single_action_space.n', envs.single_action_space.n)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # ALGO Logic: Storage setup(rollout buffer)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = torch.Tensor(envs.reset()).to(device) # initial observation ???
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # anneal learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.lr
            optimizer.param_groups[0]['lr'] = lr_now

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                value[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # run the env and log data
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = terminated or truncated
            rewards[step] = torch.tensor(reward).to(device).view(-1) # why view
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device) # Tensor???

            # for item in info:
            #     if "episode" in item.keys():
            #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         break

            # modification for new version of gymnasium
            if 'episode' in infos:
                for idx in range(args.num_envs):
                    if infos['_episode'][idx]:
                        print(f"global_step={global_step}, env {idx} finished, episodic_return={infos['episode']['r'][idx]}")
                        writer.add_scalar('episodic_return', infos['episode']['r'][idx], global_step)
                        writer.add_scalar('episodic_length', infos['episode']['l'][idx], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * next_return * nextnonterminal
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1, envs.single_observation_space.shape))
        b_actions = actions.reshape((-1, envs.single_action_space.shape))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # optimizing policy and value networks 
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.epoches):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.mini_batch_size):
                end = start + args.mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp() ## don't understand

                mb_advantages = b_advantages[mb_inds]
                

            







