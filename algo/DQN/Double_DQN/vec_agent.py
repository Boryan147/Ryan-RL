import torch
import torch.nn as nn
from algo.DQN.vec_agent import VecDQNAgent

class VecDoubleDQNAgent(VecDQNAgent):
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

        # Double DQN: decouple action selection by policy_net and evaluation by target_net
        with torch.no_grad():
            next_max_action = self.policy_net(next_state_batch).max(1).indices.unsqueeze(1)
            Q_target = self.target_net(next_state_batch).gather(1, next_max_action).squeeze(1)

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
