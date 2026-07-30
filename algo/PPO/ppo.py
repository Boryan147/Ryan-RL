import argparse
import os 
from distutils.util import strtobool
import random
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym


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
    args = parser.parse_args()
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