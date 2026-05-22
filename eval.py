import gymnasium as gym
import torch
import numpy as np
import imageio
from algo.DQN.agent import DQNAgent
from train import load_config, set_seed

def evaluate_and_generate_gif(config_path, model_path, output_gif='cartpole_demo.gif', num_episodes=3):
    """
    Load trained model and generate a GIF of the agent playing CartPole
    """
    # Load config and set seed
    config = load_config(config_path)
    env_cfg = config['env']
    net_cfg = config['network']
    hp_cfg = config['hyperparameters']
    exp_cfg = config['exploration']
    train_cfg = config['training']
    
    set_seed(train_cfg['seed'])
    
    # Initialize environment with rgb_array render mode
    env = gym.make(env_cfg['name'], render_mode='rgb_array')
    
    # Initialize agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        config={**net_cfg, **hp_cfg, **exp_cfg}
    )
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()
    
    # Collect frames
    all_frames = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        total_reward = 0
        
        while True:
            # Capture frame
            frame = env.render()
            all_frames.append(frame)
            
            with torch.no_grad():
                # Exploitation only (no exploration, epsilon=0)
                action = agent.policy_net(state).max(1).indices.view(1, 1)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            state = next_state
            
            if done:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
                break
    
    # Save as GIF
    print(f"Saving GIF to {output_gif}...")
    imageio.mimwrite(output_gif, all_frames, duration=0.05, loop=0)
    print(f"Done! GIF saved as {output_gif}")
    
    env.close()

if __name__ == "__main__":
    evaluate_and_generate_gif(
        config_path="configs/DQN_cartpole.yaml",
        model_path="results/DQN_cartpole/final_model.pth",
        output_gif="results/DQN_cartpole/cartpole_demo.gif",
        num_episodes=3
    )
