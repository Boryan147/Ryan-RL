import gymnasium as gym
import torch
import yaml
import random
import argparse
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from algo.registry import make_agent
from utils.visuals import plot_learning_curve

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 

def train(config_path):
    # Load Configurations
    config = load_config(config_path)
    env_cfg = config['env']
    train_cfg = config['training']
    algo = config['algo']
    
    # Set reproducibility seeds
    set_seed(train_cfg['seed'])

    # Initialize Environment
    env = gym.make(env_cfg['name'])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # merge all yaml sections into a single config dictionary
    agent_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            agent_config.update(value)
        else:
            agent_config[key] = value

    # Instantiate Agent 
    agent = make_agent( 
        algo,
        env.observation_space.shape[0],
        env.action_space.n,
        config=agent_config
    )

    # setup tensorboard logger
    log_dir = os.path.join(train_cfg['save_dir'], f"{env_cfg['name']}_{algo}_seed{train_cfg['seed']}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logging to: {log_dir}")

    # train the agent
    episode_rewards = agent.learn(env, writer=writer)
    plot_learning_curve(episode_rewards, save_dir=train_cfg['save_dir'], filename="final_learning_curve.png")
    
    env.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL Agent")
    parser.add_argument("--config", type=str, default="configs/DQN_cartpole.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)