import numpy as np
import random


class Agent:
    def __init__(self, n_actions, world_size, lr, y, e):
        self.n_actions = n_actions
        self.Q_table = np.zeros([world_size, self.n_actions])
        self.side_size = int(np.sqrt(world_size))

        self.lr = lr
        self.y = y
        self._e0 = e
        self.e = self._e0
        self._noise = 1.0
        self.noise = self._noise

    def _flatten(self, cord):
        return cord[0] * self.side_size + cord[1]

    def _decaying(self, episode_num):
        self.e = self._e0 ** episode_num
        self.noise = self._noise / (episode_num + 1)

    def get_action(self, episode_num, state):
        self._decaying(episode_num)

        # choose an action e-greedly
        if random.random() < self.e:
            # Exploration method called 'decaying'
            # choose an action randomly (epsilon)
            action = random.randint(0, self.n_actions - 1)
        else:
            q = self.Q_table[self._flatten(state)]

            # Exploration method called 'random noise' also using decaying
            # np.random.randn() adds gaussian random noise on Q-table for extra exploration
            q += np.random.randn(self.n_actions, ) * self.noise

            # Do not use bare np.argmax because it can not treat the case of existing multiple max values
            # But we add random noise on q, so it is fine to use argmax because there are really few chance to get same value
            action = random.choice(np.where(q == q.max())[0])

        return action

    def learn(self, state, action, reward, next_state):
        q1 = self.Q_table[self._flatten(state)][action]
        q2 = reward + self.y * np.max(self.Q_table[self._flatten(next_state)])
        self.Q_table[self._flatten(state)][action] += self.lr * (q2 - q1)
