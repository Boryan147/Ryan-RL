import gymnasium as gym
import torch
import yaml
import random
import argparse
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from algo.registry import make_agent

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

    # merge all yaml sections into a single config dictionary
    agent_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            agent_config.update(value)
        else:
            agent_config[key] = value

    # Set reproducibility seeds
    set_seed(agent_config['seed'])

    # Setup run name & logging paths
    run_name = f"{agent_config['name']}_{agent_config['algo']}_seed{agent_config['seed']}"
    log_dir = os.path.join(agent_config['save_dir'], run_name)

    # Initialize single environment
    # env = gym.make(agent_config['name'])
    # env = gym.wrappers.RecordEpisodeStatistics(env)

    # Initialize vectorized environment
    num_envs = agent_config['num_envs']
    envs = gym.make_vec(agent_config['name'], num_envs=num_envs, vectorization_mode='sync')
    envs = RecordEpisodeStatistics(envs)

    # Instantiate Agent 
    agent = make_agent( 
        agent_config['algo'],
        envs.single_observation_space.shape[0],
        envs.single_action_space.n,
        config=agent_config
    )

    # Initialize Weights & Biases if tracking is enabled
    if agent_config['track']:
        import wandb

        wandb.init(
            project=agent_config["wandb_project"],
            group=agent_config['group'],
            name=run_name,
            config=config,
            sync_tensorboard=True,               
            save_code=True,
        )
        print(f"Weights & Biases tracking initialized for project: {agent_config['wandb_project']} (group: {agent_config['group']})")

    # setup tensorboard logger
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logging to: {log_dir}")

    # train the agent
    agent.learn(envs, writer=writer)
    
    envs.close()
    writer.close()
    
    if agent_config['track']:
        import wandb
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL Agent")
    parser.add_argument("--config", type=str, default="configs/DQN_cartpole.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)