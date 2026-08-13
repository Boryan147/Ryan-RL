import torch
import torch.nn as nn
import torch.optim as optim
from algo.DQN.Dueling_DQN.dueling_net import Dueling_DQN
from algo.DQN.Double_DQN.vec_agent import VecDoubleDQNAgent

class VecDuelingAgent(VecDoubleDQNAgent):
    def __init__(self, state_size, action_size, config):
        super().__init__(state_size, action_size, config)
        # Dueling Networks
        self.policy_net = Dueling_DQN(state_size, config['hidden_size'], action_size).to(self.device)
        self.target_net = Dueling_DQN(state_size, config['hidden_size'], action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config['lr'], amsgrad=True)