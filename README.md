# 🚀 Ryan-RL: A Personal Reinforcement Learning Laboratory

Welcome to **Ryan-RL**, a clean, modular, and highly structured repository built from scratch to study, implement, and compare Reinforcement Learning (RL) algorithms.

---

## 🎯 Repository Purpose

The primary goal of this repository is to **master Reinforcement Learning from the ground up**. Rather than relying on high-level, black-box libraries (e.g. Stable-Baselines3), this repository focuses on writing readable, production-grade PyTorch implementations of classic and modern RL agents. 

This codebase serves as a framework to:
* **Deeply Understand Math-to-Code Translation:** Implement core reinforcement learning algorithms directly from their mathematical formulations and papers.
* **Explore Hyperparameter Sensitivity:** Analyze how discount factors ($\gamma$), soft target updates ($\tau$), schedules, and buffer sizes affect convergence and training stability.
* **Standardize Benchmarking:** Maintain a modular pipeline to easily compare different families of algorithms (Value-based, Policy Gradient, Actor-Critic) under identical conditions.

---

## 🗺️ Big Picture & Architecture

The repository is built around a highly modular design pattern. Every algorithm uses the same structural conventions, allowing components (like replay buffers, schedulers, and network layers) to be easily swapped.

```
Ryan-RL/
├── 📂 algo/                  # RL Agent Implementations
│   └── 📂 DQN/                # Deep Q-Network Agent & deep dive
│       ├── README.md         # 📖 DQN mathematical details, configurations, & results
│       ├── agent.py          # Action-selection, experience collection, & SGD updates
│       └── model.py          # Q-network architecture (PyTorch MLP)
├── 📂 configs/               # YAML configurations for reproducible experiments
├── 📂 utils/                 # Modular, reusable helper components
│   ├── buffers.py            # Replay buffers (Off-Policy transitions)
│   ├── schedules.py          # Exploration / Parameter schedulers
│   └── visuals.py            # Real-time training curve visualizer
├── 📂 results/               # Saved weights, logs, training plots, and demo GIFs
├── train.py                  # Standardized training pipeline entry-point
├── eval.py                   # Standardized evaluation & GIF generation entry-point
└── environment.yml           # Conda package specifications
```

### 🔍 Detailed Module Functions

| Module Component | Location | Role & Responsibility |
| :--- | :--- | :--- |
| **Algorithms (`algo/`)** | `algo/<algorithm_name>/` | Houses specific RL agent logic. It is completely isolated per algorithm, containing the mathematical state updates, target net coordination, and PyTorch model architectures. |
| **Configurations (`configs/`)** | `configs/<config_name>.yaml` | Decouples hyperparameters from code. Controls environmental setup, network sizes, exploration rates, and seeds to guarantee 100% reproducibility. |
| **Utilities (`utils/`)** | `utils/` | Reusable utilities shared across multiple agents. Includes memory structures (buffers), learning schedules, and plotting tools. |
| **Training Pipeline** | `train.py` | The main execution entry point. Parses configuration files, instantiates environments, runs the interaction loops, and triggers agent updates. |
| **Evaluation Pipeline** | `eval.py` | Loads trained models, runs deterministic inference, calculates performance metrics, and renders a `.gif` video showing the agent's behavior. |

---

## 📊 Algorithm Implementation Index

Below is a summary of the algorithms implemented or planned in this repository:

| Algorithm | Family | Action Space | Status | Documentation & Results |
| :--- | :--- | :--- | :--- | :--- |
| **DQN (Deep Q-Network)** | Value-based | Discrete | 🟢 Implemented | [DQN Deep Dive & Performance](algo/DQN/README.md) |
| **Double DQN (DDQN)** | Value-based | Discrete | 🟡 Planned | *Coming Soon* |
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

---

## 🎯 Future Comparative Benchmark Plans

Once multiple algorithms are implemented, a unified comparative suite will be developed to evaluate:
1. **Sample Efficiency:** Total environment steps required to reach target reward limits.
2. **Algorithmic Stability:** Variance in rewards across different random seed initializations.
3. **Wall-clock Speed:** Training time comparison between on-policy and off-policy algorithms.
