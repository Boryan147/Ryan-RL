# Ryan-RL: A Personal Reinforcement Learning Laboratory

Welcome to **Ryan-RL**, a clean, modular, and highly structured repository built from scratch to study, implement, and compare Reinforcement Learning (RL) algorithms. 


## Repository Purpose

The primary goal of this repository is to help me **master Reinforcement Learning from the ground up**. Rather than relying on high-level, black-box libraries (e.g. Stable-Baselines3), this repository focuses on writing readable and reproducible PyTorch implementations of classic and modern RL agents. 

Moreover, I hope this repo can be a reference for beginners who want to learn Rl algorithms with hands-on experience.

## Big Picture & Architecture

The repository is built around a highly modular design pattern. Every algorithm uses the same structural conventions, allowing components (like replay buffers, schedulers, and network layers) to be easily swapped.

```
Ryan-RL/
├── 📂 algo/                  # RL Agent Implementations
│   └── 📂 DQN/                # Deep Q-Network Agent & deep dive
│       ├── README.md         # DQN mathematical details, configurations, & results
│       ├── agent.py          # Action-selection, experience collection, & SGD updates
│       └── model.py          # Q-network architecture
├── 📂 configs/               # YAML configurations for reproducible experiments
├── 📂 utils/                 # Modular, reusable helper components
│   ├── buffers.py            # Replay buffers
│   ├── schedules.py          # Exploration / Parameter schedulers
│   └── visuals.py            # Real-time training curve visualizer
├── 📂 results/               # Saved weights, logs, training plots, and demo GIFs
├── train.py                  # Standardized training pipeline entry-point
├── eval.py                   # Standardized evaluation & GIF generation entry-point
└── environment.yml           # Conda package specifications
```

## Algorithm Implementation Index

Below is a summary of the algorithms I've implemented or planned in this repository:

| Algorithm | Family | Action Space | Status | Documentation & Results |
| :--- | :--- | :--- | :--- | :--- |
| **DQN (Deep Q-Network)** | Value-based | Discrete | 🟢 Implemented | [DQN Deep Dive & Performance](algo/DQN/README.md) |
| **Double DQN (DDQN)** | Value-based | Discrete | 🟢 Implemented | [DDQN Deep Dive & Noisy Benchmarks](algo/DQN/Double_DQN/README.md) |
| **REINFORCE** | Policy Gradient | Discrete / Continuous | 🟡 Planned | *Coming Soon* |
| **PPO (Proximal Policy)** | Actor-Critic | Discrete / Continuous | 🟡 Planned | *Coming Soon* |
| **SAC (Soft Actor-Critic)** | Actor-Critic (Max Ent) | Continuous | 🟡 Planned | *Coming Soon* |

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
