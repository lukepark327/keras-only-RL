import numpy as np
from copy import deepcopy

from utils import Enum


class Env:
    def __init__(self, n_agents: int):
        self.action_space = Enum(['Up', 'Down', 'Left', 'Right'])
        self.state_space = Enum(['Normal', 'Start', 'Goal', 'Obstacle'])

        self.world = None  # size * size
        self.size = None

        self.n_agents = n_agents
        self.agent_states = [None for _ in range(self.n_agents)]

        self.reset()

    def _create_world(self):
        size = 6
        start = (0, 0)  # cordinate
        goals = {(4, 4)}
        obstacles = {(3, 4), (4, 2), (4, 3), (4, 5)}

        # initialization
        world = np.full((size, size), self.state_space.Normal)

        # set Start point
        world[start] = self.state_space.Start

        # set Goal points
        for goal in goals:
            if world[goal] != self.state_space.Normal:
                raise Exception('Goal cannot be the same as Start or Obstacles: ', goal)
            world[goal] = self.state_space.Goal

        # set Obstacle points
        for obstacle in obstacles:
            if world[obstacle] != self.state_space.Normal:
                raise Exception('Obstacle cannot be the same as Start or Goals: ', obstacle)
            world[obstacle] = self.state_space.Obstacle

        return world, start

    def reset(self):
        self.world, state = self._create_world()
        self.size = self.world.shape[0]

        self.agent_states = [state for _ in range(self.n_agents)]
        return deepcopy(self.agent_states)  # first new observation

    def step(self, agent_num, action):
        # initialization
        next_state = None
        reward = 0
        flag = False
        info = {}  # extra info.

        # get next state
        next_state = self._take_action(
            self.agent_states[agent_num],
            agent_num,
            action)

        # get reward and flag
        if self.world[next_state] == self.state_space.Goal:
            reward = 1
            flag = True
        if self.world[next_state] == self.state_space.Obstacle:
            reward = -1
        else:
            pass  # default

        return next_state, reward, flag, info

    def _take_action(self, current_state, agent_num, action):
        if action == self.action_space.Up:
            next_state = (
                max(current_state[0] - 1, 0),
                current_state[1])
        elif action == self.action_space.Down:
            next_state = (
                min(current_state[0] + 1, self.size - 1),
                current_state[1])
        elif action == self.action_space.Left:
            next_state = (
                current_state[0],
                max(current_state[1] - 1, 0))
        elif action == self.action_space.Right:
            next_state = (
                current_state[0],
                min(current_state[1] + 1, self.size - 1))
        else:
            pass  # default

        self.agent_states[agent_num] = next_state
        return next_state

    # TODO
    def render(self):
        pass
