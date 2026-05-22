import gymnasium as gym
import torch
import yaml
import random
import argparse
import numpy as np
import os
from itertools import count
from algo.DQN.agent import DQNAgent
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
    
    # Extract specific groupings
    env_cfg = config['env']
    net_cfg = config['network']
    hp_cfg = config['hyperparameters']
    exp_cfg = config['exploration']
    train_cfg = config['training']
    
    # Set reproducibility seeds
    set_seed(train_cfg['seed'])

    # Initialize Environment
    env = gym.make(env_cfg['name'])
    
    # Initialize Your Modular Agent (passing configuration objects)
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        config={**net_cfg, **hp_cfg, **exp_cfg}
    )
    print(f"Successfully started training on {env_cfg['name']} with Learning Rate: {hp_cfg['lr']}")
    # training loop
    episode_durations = []
    for episode in range(train_cfg['n_episode'] + 1):
        episode_seed = train_cfg['seed'] + episode
        state, info = env.reset(seed=episode_seed)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        for step in count():
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            reward = torch.tensor([reward], dtype=torch.float32).to(agent.device)
            done = torch.tensor([done], dtype=bool).to(agent.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.optimize_model()
            agent.update_target_network()
            if done.item():
                episode_durations.append(step + 1)
                break
        if episode % 50 == 0:
            recent_durations = episode_durations[-50:]
            avg_duration = np.mean(recent_durations)
            print(f"Episode {episode:4d} | Last 50 Avg Duration: {avg_duration:6.2f} | Total Steps: {agent.steps_done}")
            
            # Intermediate Plotting: Updates the image file on disk in real-time
            plot_learning_curve(episode_durations, save_dir=train_cfg['save_dir'], filename="learning_curve.png")

    print("Training finished! Saving final configurations...")
    torch.save(agent.policy_net.state_dict(), os.path.join(train_cfg['save_dir'], "final_model.pth"))
    plot_learning_curve(episode_durations, save_dir=train_cfg['save_dir'], filename="final_learning_curve.png")
    
    # Save the raw reward array as well, in case you want to plot it differently later
    np.save(os.path.join(train_cfg['save_dir'], "rewards.npy"), np.array(episode_durations))
    
    env.close()

if __name__ == "__main__":
    # Command line argument parser lets you pass different configs dynamically
    parser = argparse.ArgumentParser(description="Train an RL Agent")
    parser.add_argument("--config", type=str, default="configs/DQN_cartpole.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)