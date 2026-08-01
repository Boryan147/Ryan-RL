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
    args = parser.parse_args()
    args.batch_size = int(args.num_steps * args.num_envs)
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

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":
    args = parse_args()
    run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}__{args.int(time.time())}'
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
    next_obs = torch.Tensor(envs.reset()).to(device) # initial observation
    next_done = torch.zeros(args.num_envs).to(device)
    num_updats = args.timesteps // args.batch_size
