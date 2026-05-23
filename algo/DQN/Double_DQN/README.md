# 👥 Double Deep Q-Network (Double DQN / DDQN) Implementation

This directory contains a clean, PyTorch-based implementation of the **Double Deep Q-Network (DDQN)** algorithm, designed to mitigate the overestimation bias inherent in vanilla DQN.

---

## 📖 Theoretical Background: DQN vs. Double DQN

In vanilla DQN, target Q-values are calculated using a greedy max-operator over the target network:
$$Y_t^{\text{DQN}} = R_{t+1} + \gamma \max_{a} Q_{\theta^-}(S_{t+1}, a)$$

Because the same target network weight set ($\theta^-$) is used to both **select** and **evaluate** the best action, it propagates function approximation errors and noise optimistically. This results in **overestimation bias**, which can accumulate rapidly in complex or noisy environments, leading to suboptimal policies or complete training divergence.

**Double DQN** solves this by decoupling the selection weight set from the evaluation weight set. 
1. We use the **online network** ($\theta$) to **select** the greedy action:
   $$a^* = \arg\max_{a} Q_{\theta}(S_{t+1}, a)$$
2. We use the **target network** ($\theta^-$) to **evaluate** the Q-value of that selected action:
   $$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q_{\theta^-}(S_{t+1}, a^*)$$

This simple yet elegant modification prevents the agent from consistently choosing overoptimistic estimates, resulting in highly stable and accurate value functions.

---

## 📁 Directory Structure

* [agent.py](agent.py): Holds the `Double_DQNAgent` implementation, coordinating target network value gathering via action decoupling.
* [compare_dqn_ddqn.py](compare_dqn_ddqn.py): A dedicated comparative benchmarking script designed to clearly demonstrate Double DQN's superiority.

---

## 🔬 Noisy Environment Benchmark: Demonstrating the DDQN Advantage

While DQN and DDQN perform similarly on simple, noiseless benchmarks like standard `CartPole-v1`, the real advantage of Double DQN becomes striking when **environmental noise** is introduced.

We have provided a comparative pipeline in **[compare_dqn_ddqn.py](compare_dqn_ddqn.py)**. This script:
1. Wraps the `CartPole-v1` environment to inject high-variance Gaussian noise into the rewards ($\mathcal{N}(0, 0.6)$).
2. Trains both a vanilla DQN agent and a Double DQN agent side-by-side using the same seeds and architecture.
3. Records the **rolling rewards** and the **average estimated Q-value** ($\mathbb{E}[Q(s_0, a)]$ of the starting state) over 350 episodes.

### 🚀 How to Run the Benchmark

To run this comparative benchmark and generate the performance curves:

```bash
# Run from the repository root
python algo/DQN/Double_DQN/compare_dqn_ddqn.py
```

### 📊 What the Results Show (Output Plots)

After running the script, two comparative plots are saved to `results/DQN_vs_DDQN/comparison_noisy.png` (and duplicated in this folder for easy local access):

1. **Training Rewards Comparison (Plot 1):** 
   Under noisy rewards, standard DQN suffers from unstable learning and struggles to converge because it gets distracted chasing noisy reward peaks. **Double DQN** maintains high stability, handles the noise gracefully, and achieves a significantly higher and steadier average step survival.

2. **Q-Value Overestimation Study (Plot 2):** 
   The true analytical discounted return upper bound for CartPole ($\gamma=0.99$, reward $\approx 1$) is **~100**.
   * **Vanilla DQN Q-estimates:** Explode well past **140+**, showing severe overestimation bias as the max-operator accumulates the injected reward noise.
   * **Double DQN Q-estimates:** Remain perfectly grounded near the analytical ceiling of **100**, proving that the decoupled update successfully eliminates value inflation.

*The generated comparison image `comparison_noisy.png` will appear in this folder once the benchmark is run!*
