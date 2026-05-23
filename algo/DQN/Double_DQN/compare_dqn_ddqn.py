import sys
import os
import random
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count

# Ensure project root is in the Python search path to allow relative/absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from algo.DQN.agent import DQNAgent
from algo.DQN.Double_DQN.agent import Double_DQNAgent

# =====================================================================
# 1. Custom Noisy Reward Wrapper
# =====================================================================
class NoisyRewardWrapper(gym.RewardWrapper):
    """
    Gymnasium wrapper that adds Gaussian noise to rewards.
    This injects estimation noise, which triggers severe overestimation bias
    in vanilla DQN, while Double DQN remains stable and robust.
    """
    def __init__(self, env, noise_std=0.5):
        super().__init__(env)
        self.noise_std = noise_std

    def reward(self, reward):
        # Add normal random noise to the reward
        return reward + np.random.normal(0, self.noise_std)

# =====================================================================
# 2. Reproduction Seeds & Hyperparameters
# =====================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Benchmark configurations
CONFIG = {
    'hidden_size': 128,
    'lr': 1.0e-3,           # Slightly higher learning rate to accentuate noise sensitivity
    'gamma': 0.99,
    'tau': 0.005,
    'buffer_capacity': 10000,
    'batch_size': 128,
    'max_epsilon': 0.9,
    'min_epsilon': 0.01,
    'decay_rate': 1500,     # Faster decay rate for quick convergence demonstration
    'seed': 42
}

N_EPISODES = 350
NOISE_STD = 0.6  # Standard deviation of reward noise

# =====================================================================
# 3. Training Function
# =====================================================================
def train_agent(agent_type="DQN"):
    set_seed(CONFIG['seed'])
    
    # Create the noisy environment
    raw_env = gym.make('CartPole-v1')
    env = NoisyRewardWrapper(raw_env, noise_std=NOISE_STD)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    if agent_type == "DQN":
        agent = DQNAgent(state_size, action_size, CONFIG)
    else:
        agent = Double_DQNAgent(state_size, action_size, CONFIG)
        
    print(f"\n--- Training {agent_type} on Noisy CartPole-v1 (Reward Noise Std: {NOISE_STD}) ---")
    
    episode_durations = []
    initial_q_values = []
    
    for episode in range(N_EPISODES):
        episode_seed = CONFIG['seed'] + episode
        state, info = env.reset(seed=episode_seed)
        
        # Keep track of initial state Q-value estimate to measure overestimation
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_est = agent.policy_net(state_tensor).max(1).values.item()
            initial_q_values.append(q_est)
            
        for step in count():
            action = agent.select_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(agent.device)
            done_tensor = torch.tensor([done], dtype=bool).to(agent.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            
            agent.memory.push(state_tensor, action, reward_tensor, next_state_tensor, done_tensor)
            state_tensor = next_state_tensor
            
            agent.optimize_model()
            agent.update_target_network()
            
            if done:
                episode_durations.append(step + 1)
                break
                
        if (episode + 1) % 50 == 0:
            avg_rew = np.mean(episode_durations[-50:])
            avg_q = np.mean(initial_q_values[-50:])
            print(f"Episode {episode+1:3d} | Last 50 Avg Steps: {avg_rew:6.2f} | Avg predicted Q_0: {avg_q:6.2f}")
            
    env.close()
    return episode_durations, initial_q_values

# =====================================================================
# 4. Run Benchmark & Plot Results
# =====================================================================
def run_benchmark():
    # 1. Train Vanilla DQN
    dqn_rewards, dqn_q_vals = train_agent("DQN")
    
    # 2. Train Double DQN
    ddqn_rewards, ddqn_q_vals = train_agent("Double DQN")
    
    # Create the output directory (use absolute path based on project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    save_dir = os.path.join(project_root, 'results', 'DQN_vs_DDQN')
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate rolling averages for plotting
    window = 30
    def rolling_avg(data):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Align x-axes for rolling averages
    x_axis_rolling = np.arange(window - 1, N_EPISODES)
    
    print("\nPlotting comparative performance and Q-value overestimation study...")
    
    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ------------------ Plot 1: Episode Duration / Reward ------------------
    ax1.set_title("1. Training Progress on Noisy CartPole-v1", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel(f"Steps survived ({window}-Ep Moving Avg)", fontsize=11)
    
    ax1.plot(x_axis_rolling, rolling_avg(dqn_rewards), label="Vanilla DQN", color="#e74c3c", linewidth=2.5)
    ax1.plot(x_axis_rolling, rolling_avg(ddqn_rewards), label="Double DQN", color="#3498db", linewidth=2.5)
    
    # Add raw markers transparently in background
    ax1.plot(dqn_rewards, alpha=0.15, color="#e74c3c")
    ax1.plot(ddqn_rewards, alpha=0.15, color="#3498db")
    
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # ------------------ Plot 2: Q-Value Overestimation ------------------
    ax2.set_title("2. Estimated Starting State Q-Value", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Predicted Q(s_0, a)", fontsize=11)
    
    # True maximum discounted reward is ~100 under gamma=0.99
    # (Since maximum steps = 500, sum_{t=0}^{500} 0.99^t is bounded by 100)
    ax2.axhline(y=100.0, color="#2ecc71", linestyle="--", linewidth=1.5, label="True Analytical Upper Bound (~100)")
    
    ax2.plot(x_axis_rolling, rolling_avg(dqn_q_vals), label="Vanilla DQN Q-Est", color="#e74c3c", linewidth=2.5)
    ax2.plot(x_axis_rolling, rolling_avg(ddqn_q_vals), label="Double DQN Q-Est", color="#3498db", linewidth=2.5)
    
    # Add raw markers transparently in background
    ax2.plot(dqn_q_vals, alpha=0.15, color="#e74c3c")
    ax2.plot(ddqn_q_vals, alpha=0.15, color="#3498db")
    
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    # Save files in both results and local directory for easy previewing
    fig_path = os.path.join(save_dir, "comparison_noisy.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig("./comparison_noisy.png", dpi=300, bbox_inches='tight')
    
    print(f"\nDone! Comparative plots saved successfully to:")
    print(f" 1. {os.path.abspath(fig_path)}")
    print(f" 2. {os.path.abspath('./comparison_noisy.png')}")
    print("\nObservations to check in the plots:")
    print(" - Vanilla DQN's Q-values explode way above the ~100 theoretical upper limit (Overestimation Bias).")
    print(" - Double DQN's Q-values stay stably grounded near the true analytical ceiling.")
    print(" - Due to this grounding, Double DQN achieves a higher, more stable average score under noisy rewards.")

if __name__ == "__main__":
    run_benchmark()
