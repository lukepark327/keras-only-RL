import numpy as np
from collections import namedtuple


class Env:
    def __init__(self):
        Actions = namedtuple('Actions', ['Up', 'Down', 'Left', 'Right'])
        States = namedtuple('States', ['Normal', 'Start', 'Goal', 'Obstacle'])
        self.actions = Actions(0, 1, 2, 3)
        self.states = States(0, 1, 2, 3)

        self.world = None
        self.side_size = None

        self.agent_state = None  # only one agent

        self.reset()

    def _create_world(self):
        side_size = 6  # N
        start = (0, 0)  # cordinate
        goals = {(4, 4)}
        obstacles = {(3, 4), (4, 2), (4, 3), (4, 5)}

        # initialization
        world = np.full((side_size, side_size), self.states.Normal)  # (N, N)

        # set Start point
        world[start] = self.states.Start

        # set Goal points
        for goal in goals:
            if world[goal] != self.states.Normal:
                raise Exception('Goal cannot be the same as Start or Obstacles: ', goal)
            world[goal] = self.states.Goal

        # set Obstacle points
        for obstacle in obstacles:
            if world[obstacle] != self.states.Normal:
                raise Exception('Obstacle cannot be the same as Start or Goals: ', obstacle)
            world[obstacle] = self.states.Obstacle

        return world, start

    def reset(self):
        self.world, init_state = self._create_world()
        self.side_size = self.world.shape[0]

        self.agent_state = init_state
        return self.agent_state

    def _take_action(self, current_state, action):
        if action == self.actions.Up:
            next_state = (
                max(current_state[0] - 1, 0),
                current_state[1])
        elif action == self.actions.Down:
            next_state = (
                min(current_state[0] + 1, self.side_size - 1),
                current_state[1])
        elif action == self.actions.Left:
            next_state = (
                current_state[0],
                max(current_state[1] - 1, 0))
        elif action == self.actions.Right:
            next_state = (
                current_state[0],
                min(current_state[1] + 1, self.side_size - 1))
        else:
            pass  # default

        self.agent_state = next_state
        return next_state

    def step(self, action):
        # initialization
        next_state = None
        reward = 0
        flag = False
        info = {}  # extra info.

        # get next state
        next_state = self._take_action(self.agent_state, action)

        # get reward and flag
        if self.world[next_state] == self.states.Goal:
            reward = 1
            flag = True
        if self.world[next_state] == self.states.Obstacle:
            reward = -1
        else:
            pass  # default

        return next_state, reward, flag, info

    # TODO
    def render(self):
        pass
