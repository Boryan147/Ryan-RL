from .DQN.agent import DQNAgent
from .DQN.vec_agent import VecDQNAgent
from .DQN.Double_DQN.doubledqn_agent import Double_DQNAgent
from .DQN.Dueling_DQN.dueling_agent import Dueling_DQNAgent
from .REINFORCE.agent import REINFORCEAgent

# register all the possible algorithm agents
Agent_Registry = {
    'DQN': DQNAgent,
    'VecDQN': VecDQNAgent,
    'DoubleDQN': Double_DQNAgent,
    'DuelingDQN': Dueling_DQNAgent,
    'REINFORCE': REINFORCEAgent
}

# instantiate agent
def make_agent(algo, obs_size, act_size, config):
    if algo not in Agent_Registry:
        raise ValueError(f'Algorithm {algo} not found. Choose from {list(Agent_Registry.keys())}')
    return Agent_Registry[algo](obs_size, act_size, config)