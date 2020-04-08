from collections import namedtuple, deque
import random


class ReplayMemory:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
        self.Experience = namedtuple('Experience', ['State', 'Action', 'Reward', 'Next_state'])

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, reward, next_state):
        e = self.Experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)
