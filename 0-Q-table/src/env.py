import numpy as np
from copy import deepcopy

from utils import Enum, Cord


class State_type(Enum):
    def __init__(self, state_types: list):
        Enum.__init__(self, state_types)


class Env:
    def __init__(
            self,
            state_types: State_type,
            states: np.array,
            s0: Cord):

        self.state_types = state_types
        self.states = states
        self.size = self.states.shape[0]
        self.__s0 = s0  # base(init) state
        self.__s = deepcopy(self.s0)  # s: current state

    @property  # getter
    def s0(self):
        return self.__s0

    @property  # getter
    def s(self):
        return self.__s

    @s.setter  # setter
    def s(self, new_s: Cord):
        self.__s = deepcopy(new_s)

    def reset(self):
        self.s = self.s0
        return self.s.flatten_index(self.size)

    # TODO: visualization via image
