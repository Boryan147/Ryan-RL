import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .model import DQN
from utils.buffers import Replaybuffer
from utils.schedules import EpsilonGreedy

class DQNAgent:
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

    def select_action(self, state):
        sample = random.random()
        epsilon_threshold = self.epsilon_greedy.get_epsilon(self.steps_done)
        self.steps_done += 1
        if sample > epsilon_threshold:
            with torch.no_grad():
                # state shape = [1, state_size]
                return self.policy_net(state).max(1).indices.view(1,1).to(self.device)
        else:
            return torch.tensor([[random.randint(0, self.action_size - 1)]]).to(self.device)
    
    def optimize_model(self):
        if len(self.memory) < self.config['batch_size']:
            return
        # randomly select a batch
        batch = self.memory.sample(self.config['batch_size'])
        # extrach state, action, reward, next_state, done from batch
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
        
        # define Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_values, TD_target.unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        # soft update
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = self.config['tau'] * policy_net_state_dict[key] + (1 - self.config['tau']) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    # agent training loop
    def learn(self, env, writer=None):
        # train_cfg = self.config['training'] 
        episode_reward = []
        global_step = 0

        for episode in range(self.config['n_episode'] + 1):
            obs, info = env.reset(seed=self.config['seed'] + episode)
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # obs shape: (1, T)

            while True:
                global_step += 1
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                # push transition to replay buffer
                t_reward = torch.tensor([reward], device=self.device)
                t_next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                t_done = torch.tensor([terminated], dtype=bool, device=self.device)

                self.memory.push(obs, action, t_reward, t_next_obs, t_done)
                obs = t_next_obs

                # update the network
                loss = self.optimize_model()
                self.update_target_network()

                # log data
                if loss is not None and writer:
                    writer.add_scalar('loss/train', loss, global_step)

                if done:
                    if 'episode' in info:
                        epi_return = info['episode']['r']
                        epi_length = info['episode']['l']
                        episode_reward.append(epi_return)
                        if writer:
                            writer.add_scalar('episodic reward', epi_return, global_step)
                            writer.add_scalar('episodic length', epi_length, global_step)
                    break
            if episode % 50 == 0 and episode_reward:
                avg_rew = np.mean(episode_reward[-50:])
                print(f"Episode {episode:4d} | Last 50 Avg rewards: {avg_rew:6.2f}")

        return episode_reward

