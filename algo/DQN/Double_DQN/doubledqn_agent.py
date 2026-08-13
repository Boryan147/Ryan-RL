import torch
import torch.nn as nn
from algo.DQN.agent import DQNAgent

class Double_DQNAgent(DQNAgent): 
    def optimize_model(self):
        if len(self.memory) < self.config['batch_size']:
            return None, None
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

        # decouple action selection & evaluation in next_state
        with torch.no_grad():
            # select max action by policy_net
            next_max_action = self.policy_net(next_state_batch).max(1).indices.unsqueeze(1)
            # evaluate the Q-value by target_net
            Q_target = self.target_net(next_state_batch).gather(1, next_max_action).squeeze(1)
        # TD target
        TD_target = reward_batch + self.config['gamma'] * Q_target * (~done_batch)

        # define Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_values, TD_target.unsqueeze(1))

        # calculate mean Q-value
        mean_q = Q_values.mean().item()

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item(), mean_q