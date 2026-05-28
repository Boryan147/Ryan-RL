# Double Deep Q-Network (Double DQN) Implementation

This directory contains an implementation of the **Double Deep Q-Network (DDQN)** algorithm, designed to mitigate the overestimation bias inherent in vanilla DQN.

## Theoretical Background: DQN vs. Double DQN

In vanilla DQN, target Q-values are calculated using a greedy max-operator over the target network:
$$
Y^{DQN}_{t}=R_{t+1}+\gamma\max\limits_{a'}Q(S_{t+1},a';\theta^-_{t})
$$

Because the same target network weight set ($\theta^-$) is used to both **select** and **evaluate** the best action, it propagates function approximation errors and noise optimistically. This results in **overestimation bias**, which can accumulate rapidly in complex or noisy environments, leading to suboptimal policies or complete training divergence.

**Double DQN** solves this by decoupling the selection weight set from the evaluation weight set. 
1. We use the **online network** ($\theta$) to **select** the greedy action:
   $$a^* = \arg\max_{a} Q(S_{t+1}, a;\theta)$$
2. We use the **target network** ($\theta^-$) to **evaluate** the Q-value of that selected action:
   $$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q(S_{t+1}, a^*;\theta^-)$$

This simple yet elegant modification prevents the agent from consistently choosing overoptimistic estimates, resulting in highly stable and accurate value functions.

## Directory Structure

* [agent.py](agent.py): Holds the `Double_DQNAgent` implementation, coordinating target network value gathering via action decoupling.
* [compare_dqn_ddqn.py](compare_dqn_ddqn.py): A dedicated comparative benchmarking script designed to clearly demonstrate Double DQN's superiority.
