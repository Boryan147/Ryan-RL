# Ryan-RL: A Personal Reinforcement Learning Laboratory

Welcome to **Ryan-RL**, a clean, modular, and highly structured repository built from scratch to study, implement, and compare Reinforcement Learning (RL) algorithms. 


## Repository Purpose

The primary goal of this repository is to help me **master Reinforcement Learning from the ground up**. Rather than relying on high-level, black-box libraries (e.g. Stable-Baselines3), this repository focuses on writing readable and reproducible PyTorch implementations of classic and modern RL agents. 

Moreover, I hope this repo can be a reference for beginners who want to learn Rl algorithms with hands-on experience.

## Big Picture & Architecture

The repository is built around a clean, modular design. Algorithms are housed in self-contained folders within `algo/`, while shared utilities reside in `utils/`:

```
Ryan-RL/
├── algo/        # RL agent implementations (DQN, REINFORCE, A2C, PPO)
├── configs/     # YAML configuration files for reproducible runs
├── utils/       # Shared helpers (replay buffers, schedules, visualizers)
├── results/     # Saved training weights, logs, and evaluation figures
├── train.py     # Training execution script
└── eval.py      # Trained agent evaluation and rendering
```

## Algorithm Implementation Index

Below is a summary of the reinforcement learning algorithms implemented in this repository:

| Algorithm | Family | Action Space | Implementation & Documentation |
| :--- | :--- | :--- | :--- |
| **DQN (Deep Q-Network)** | Value-based | Discrete | [DQN Deep Dive & Performance](algo/DQN/README.md) |
| **Double DQN (DDQN)** | Value-based | Discrete | [DDQN Deep Dive & Noisy Benchmarks](algo/DQN/Double_DQN/README.md) |
| **Dueling DQN** | Value-based | Discrete | [Dueling DQN Implementation](algo/DQN/Dueling_DQN/) |
| **REINFORCE** | Policy Gradient | Discrete | [REINFORCE Agent](algo/REINFORCE/agent.py) |
| **A2C (Advantage Actor-Critic)** | Actor-Critic | Discrete | [A2C Notebooks (Online, n-step, GAE)](algo/A2C/) |
| **PPO (Proximal Policy Optimization)** | Actor-Critic | Discrete | [PPO Code](algo/PPO/ppo.py) |

---

## 🛠️ Quick Start

### 1. Installation
Set up the Conda environment using the provided package specification:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate Ryan-RL
```

### 2. General Training Loop
Any algorithm can be trained by passing its corresponding config file:

```bash
# Example: Train the DQN agent on CartPole
python train.py --config configs/DQN_cartpole.yaml
```

### 3. General Evaluation Loop
Evaluate your trained agents and render play-through demonstrations:

```bash
# Runs evaluation and saves a play-through GIF to the results folder
python eval.py
```
