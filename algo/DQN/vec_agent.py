import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .model import DQN
from utils.buffers import Replaybuffer
from utils.schedules import EpsilonGreedy

class VecDQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

        # Networks
        self.policy_net = DQN(state_size, config['hidden_size'], action_size).to(self.device)
        self.target_net = DQN(state_size, config['hidden_size'], action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config['lr'], amsgrad=True)
        self.memory = Replaybuffer(config['buffer_capacity'])
        self.steps_done = 0
        
        # Initialize epsilon-greedy schedule once
        self.epsilon_greedy = EpsilonGreedy(
            max_epsilon=self.config['max_epsilon'],
            min_epsilon=self.config['min_epsilon'],
            decay_rate=self.config['decay_rate']
        )

    def select_action(self, obs_tensor):
        # in vectorized env, obs shape: (num_envs, obs_size)
        num_envs = obs_tensor.shape[0]
        epsilon_threshold = self.epsilon_greedy.get_epsilon(self.steps_done)
        self.steps_done += num_envs
        
        actions = []
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            
        for i in range(num_envs):
            if random.random() > epsilon_threshold:
                actions.append(greedy_actions[i])
            else:
                actions.append(random.randint(0, self.action_size - 1))
                
        return np.array(actions) # action shape (4,)
    
    def optimize_model(self):
        if len(self.memory) < self.config['batch_size']:
            return None, None
        # randomly select a batch
        batch = self.memory.sample(self.config['batch_size'])
        state_batch = torch.cat([t.state for t in batch]).to(self.device)
        action_batch = torch.cat([t.action for t in batch]).to(self.device)
        reward_batch = torch.cat([t.reward for t in batch]).to(self.device)
        next_state_batch = torch.cat([t.next_state for t in batch]).to(self.device)
        done_batch = torch.cat([t.done for t in batch]).to(self.device)

        # Q(s_t, a)
        Q_values = self.policy_net(state_batch).gather(1, action_batch)
        # maxQ(s_t+1, a')
        with torch.no_grad():
            Q_target = self.target_net(next_state_batch).max(1).values
        # TD target
        TD_target = reward_batch + self.config['gamma'] * Q_target * (~done_batch)
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_values, TD_target.unsqueeze(1))
        mean_q = Q_values.mean().item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item(), mean_q

    def update_target_network(self):
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = self.config['tau'] * policy_net_state_dict[key] + (1 - self.config['tau']) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    # Vectorized training loop
    def learn(self, envs, writer=None):
        global_step = 0
        episode_reward = []
        num_envs = envs.num_envs
        total_timesteps = self.config['total_timesteps']
        last_printed_step = 0

        obs, infos = envs.reset(seed=self.config['seed'])

        while global_step < total_timesteps:
            global_step += num_envs
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

            actions = self.select_action(obs_tensor)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # push transition for each parallel env to replay buffer
            for i in range(num_envs):
                s_i = obs_tensor[i:i+1]
                a_i = torch.tensor([[actions[i]]], device=self.device)
                r_i = torch.tensor([rewards[i]], dtype=torch.float32, device=self.device)
                ns_i = torch.tensor(next_obs[i:i+1], dtype=torch.float32).to(self.device)
                d_i = torch.tensor([terminations[i]], dtype=torch.bool, device=self.device)
                self.memory.push(s_i, a_i, r_i, ns_i, d_i)

            obs = next_obs

            # update network
            res = self.optimize_model()
            loss, q_val = res if res is not None else (None, None)
            self.update_target_network()

            # log data
            if loss is not None and writer:
                writer.add_scalar('losses/dqn_loss', loss, global_step)
                writer.add_scalar('charts/q_values', q_val, global_step)

            if '_episode' in infos:
                for i in range(num_envs):
                    if infos['_episode'][i]:
                        epi_return = infos['episode']['r'][i]
                        epi_length = infos['episode']['l'][i]
                        episode_reward.append(epi_return)
                        if writer:
                            writer.add_scalar('charts/episodic reward', epi_return, global_step)
                            writer.add_scalar('charts/episodic length', epi_length, global_step)

            # Print training progress every 10,000 global steps
            if global_step - last_printed_step >= 10000 and len(episode_reward) > 0:
                last_printed_step = global_step
                avg_rew = np.mean(episode_reward[-50:])
                print(f"Step {global_step:7d}/{total_timesteps} | Episodes: {len(episode_reward):4d} | Last 50 Avg Reward: {avg_rew:6.2f}")

