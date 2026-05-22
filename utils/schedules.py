import numpy as np

class EpsilonGreedy:
    def __init__(self, max_epsilon, min_epsilon, decay_rate):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def get_epsilon(self, steps_done):
        # exponential decay of epsilon
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-1.0 * steps_done / self.decay_rate)

        