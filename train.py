import gymnasium as gym
import torch
import yaml
import random
import argparse
import numpy as np
import os
from itertools import count
from algo.DQN.agent import DQNAgent
from algo.REINFORCE.agent import REINFORCEAgent
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
    # exp_cfg = config['exploration']
    train_cfg = config['training']
    
    # Set reproducibility seeds
    set_seed(train_cfg['seed'])

    # Initialize Environment
    env = gym.make(env_cfg['name'])
    
    # Initialize Agent 
    agent = REINFORCEAgent(
        obs_size=env.observation_space.shape[0],
        act_size=env.action_space.n,
        config={**net_cfg, **hp_cfg}
    )
    print(f"Successfully started training on {env_cfg['name']}")
    # training loop
    episode_rewards = []
    for episode in range(train_cfg['n_episode'] + 1):
        episode_seed = train_cfg['seed'] + episode
        state, _ = env.reset(seed=episode_seed)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device) # state shape: (1, state_size)

        for step in count():
            # select an action given a state
            action = agent.select_action(state)
            # move to next state
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # store episode reward 
            agent.store_rew(reward)
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            # optimize at the end of episode
            if done:
                episode_rewards.append(sum(agent.epi_rewards))
                agent.optimize()
                break

        if episode % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            avg_rewards = np.mean(recent_rewards)
            print(f"Episode {episode:4d} | Last 50 Avg rewards: {avg_rewards:6.2f}")
            
    print("Training finished! Saving final configurations...")
    # torch.save(agent.policy_net.state_dict(), os.path.join(train_cfg['save_dir'], "final_model.pth"))
    plot_learning_curve(episode_rewards, save_dir=train_cfg['save_dir'], filename="final_learning_curve.png")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL Agent")
    parser.add_argument("--config", type=str, default="configs/reinforce_cartpole.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)