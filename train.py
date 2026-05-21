import gymnasium as gym
import torch
import yaml
import random
import argparse
import numpy as np
from algo.DQN.agent import DQNAgent
from utils.schedules import EpsilonGreedy

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
    # 1. Load Configurations
    config = load_config(config_path)
    
    # 2. Extract specific groupings
    env_cfg = config['env']
    hp_cfg = config['hyperparameters']
    exp_cfg = config['exploration']
    train_cfg = config['training']
    
    # Set reproducibility seeds
    set_seed(train_cfg['seed'])

    # 3. Initialize Environment
    env = gym.make(env_cfg['id'])
    
    # 4. Initialize Your Modular Agent (passing configuration objects)
    # agent = DQNAgent(
    #     state_size=env.observation_space.shape[0],
    #     action_size=env.action_space.n,
    #     lr=hp_cfg['lr'],
    #     gamma=hp_cfg['gamma'],
    #     hidden_size=config['network']['hidden_size']
    # )
    
    print(f"Successfully started training on {env_cfg['id']} with Learning Rate: {hp_cfg['lr']}")
    # ... Rest of your training loop using the structured variables ...

if __name__ == "__main__":
    # Command line argument parser lets you pass different configs dynamically
    parser = argparse.ArgumentParser(description="Train an RL Agent")
    parser.add_argument("--config", type=str, default="configs/dqn_cartpole.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)