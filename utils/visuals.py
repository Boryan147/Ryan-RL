import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curve(episode_durations, save_dir, filename):
    """
    Plots the training durations and a rolling 50-episode average
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # Plot raw data
    durations = np.array(episode_durations)
    plt.plot(durations, alpha=0.3, label="Raw Episode Reward", color="blue")
    
    # Plot rolling 50-episode average if we have enough data
    if len(durations) >= 50:
        # Calculate moving average
        rolling_avg = np.convolve(durations, np.ones(50)/50, mode='valid')
        # Offset x-axis so the line aligns correctly with the episodes
        plt.plot(np.arange(49, len(durations)), rolling_avg, label="50-Ep Moving Avg", color="red", linewidth=2)
    elif len(durations) > 0:
        # If less than 50 episodes, show a progressive rolling average
        progressive_avg = [np.mean(durations[:i+1]) for i in range(len(durations))]
        plt.plot(progressive_avg, label="Progressive Avg", color="red", linestyle="--")

    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close() # Free up memory