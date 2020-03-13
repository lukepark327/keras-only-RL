import numpy as np
import random

from utils import Enum


class Action(Enum):
    def __init__(self, actions: list):
        Enum.__init__(self, actions)


class Agent:
    def __init__(
            self,
            actions: Action,
            x: float):

        self.actions = actions
        self.__x0 = x
        self.__x = self.x0

    @property  # getter
    def x0(self):
        return self.__x0

    @property  # getter
    def x(self):
        return self.__x

    @x.setter  # setter
    def x(self, new_x: float):
        self.__x = new_x

    def x_update(self):
        self.x *= self.x0

    def do(self, q: np.array, i: int):
        # choose an action e-greedly with Q-table
        if random.random() < self.x:
            # Exploration method called 'decaying'
            # choose an action randomly (epsilon)
            a = random.randint(0, self.actions.size - 1)
        else:
            # np.argmax returns an index which has the maximun value into the flattened array
            # np.unravel_index returns the position (row and col) from original array. For example:
            # np.unravel_index(8, np.array(L).shape)  # where 8 is return value of argmax, L is the original array
            # np.random.randn() adds gaussian random noise on Q-table for extra exploration
            # Exploration method called 'random noise' also using decaying
            q += np.random.randn(self.actions.size, ) * (1.0 / (i + 1))
            # pprint(q)

            # Do not use bare np.argmax because it can not treat the case of existing multiple max values
            # But we add random noise on q, so it is fine to use argmax because there are really few chance to get same value
            a = random.choice(np.where(q == q.max())[0])
            # pprint(a)

        return a
