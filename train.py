import gymnasium as gym
import torch
import yaml
import random
import argparse
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
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
    env_cfg = config['env']
    train_cfg = config['training']
    logging_cfg = config['logging']
    algo = config['algo']
    
    # Read tracking and video recording settings from YAML
    use_wandb = logging_cfg['track']
    wandb_project = logging_cfg["wandb_project"]
    record_video = logging_cfg["capture_video"]
    video_freq = logging_cfg["video_trigger"]
    group_name = logging_cfg['group']

    # Set reproducibility seeds
    set_seed(train_cfg['seed'])

    # Setup run name & logging paths
    run_name = f"{env_cfg['name']}_{algo}_seed{train_cfg['seed']}"
    log_dir = os.path.join(train_cfg['save_dir'], run_name)

    # Initialize Environment
    render_mode = "rgb_array" if record_video else None
    env = gym.make(env_cfg['name'], render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    if record_video:
        video_dir = os.path.join(train_cfg['save_dir'], "videos", run_name)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep_id: ep_id % video_freq == 0
        )
        print(f"Video recording enabled! Saving to: {video_dir} (every {video_freq} episodes)")

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

    # Initialize Weights & Biases if tracking is enabled
    if use_wandb:
        import wandb

        wandb.init(
            project=wandb_project,
            group=group_name,
            name=run_name,
            config=config,
            sync_tensorboard=True,               
            save_code=True,
        )
        print(f"Weights & Biases tracking initialized for project: {wandb_project} (group: {group_name})")

    # setup tensorboard logger
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logging to: {log_dir}")

    # train the agent
    agent.learn(env, writer=writer)
    
    env.close()
    writer.close()
    
    if use_wandb:
        import wandb
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL Agent")
    parser.add_argument("--config", type=str, default="configs/DQN_cartpole.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)