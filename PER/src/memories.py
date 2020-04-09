import numpy as np
from collections import namedtuple, deque
import random


class PrioritizedReplayMemory:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
        self.Experience = namedtuple('Experience', ['State', 'Action', 'Reward', 'Next_state', 'Done'])

        self.priority = deque(maxlen=buffer_size)
        self.max_priority = 1.0

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, reward, next_state, done, p=None):
        if p == None:
            p = self.max_priority  # maximal priority

        e = self.Experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priority.append(p)

    def sample(self, batch_size, beta):
        p_sum = np.sum(self.priority)
        prob = self.priority / p_sum

        indices = random.choices(range(len(prob)), k=batch_size, weights=prob)
        samples = np.array(self.memory)[indices]

        w = (len(self.priority) * prob) ** (-beta)
        importances = np.array(w)[indices]
        importances /= max(importances)  # normalization

        return indices, samples, importances

    def update_priority(self, idx, p):
        self.priority[idx] = p
