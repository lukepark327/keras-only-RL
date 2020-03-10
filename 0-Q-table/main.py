# TODO: modulization
import numpy as np
import random
import copy
from pprint import pprint
from matplotlib import pyplot as plt


class Cord:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    # TODO: changing with function named like 'as_tuple', 'tuple', et al.
    def __call__(self):
        return (self.x, self.y)

    def __repr__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'


# TODO: implement meta class for State_type and Action (same format)
class State_type:
    def __init__(self, state_types: list):
        self.__state_types = state_types
        self.size = len(state_types)
        for i, e in enumerate(state_types):
            setattr(self, e, i)

    # operator overloading
    def __getitem__(self, key):
        return self.__state_types[key]

    # define print format
    def __repr__(self):
        return '[' + ', '.join(self.__state_types) + ']'


class Action:
    def __init__(self, actions: list):
        self.__actions = actions
        self.size = len(actions)
        for i, e in enumerate(actions):
            setattr(self, e, i)

    def __getitem__(self, key):
        return self.__actions[key]

    def __repr__(self):
        return '[' + ', '.join(self.__actions) + ']'


# TODO: generalization
# TODO: print
class Env:
    def __init__(
            self,
            state_types: State_type,
            states: np.array,
            actions: np.array,
            s0: Cord = Cord(0, 0)):
        self.__state_types = state_types
        self.__states = states
        self.__size = self.__states.shape[0]
        self.__actions = actions
        self.__s0 = s0  # base(init) state
        self.s = copy.deepcopy(self.s0)  # current state

    # getter
    @property
    def s0(self):
        return self.__s0

    # # setter
    # @s0.setter
    # def s0(self, new_s0: int = 0):
    #     self.__s0 = new_s0

    def flatten_index(self, cord: Cord):
        return cord.x * self.__size + cord.y  # index of flatted array

    def reset(self):
        self.s = copy.deepcopy(self.s0)
        return self.flatten_index(self.s)

    # TODO: allowing user defined 'step' funtion
    # TODO: generalization
    def step(self, a):
        # initialization
        next_state = Cord(0, 0)
        reward = 0
        flag = False
        info = []  # extra info.

        # get next state
        if a == self.__actions.Up:
            next_state = Cord(max(self.s.x - 1, 0), self.s.y)
        elif a == self.__actions.Down:
            next_state = Cord(min(self.s.x + 1, self.__size - 1), self.s.y)
        elif a == self.__actions.Left:
            next_state = Cord(self.s.x, max(self.s.y - 1, 0))
        elif a == self.__actions.Right:
            next_state = Cord(self.s.x, min(self.s.y + 1, self.__size - 1))
        else:
            pass  # default

        # get reward and flag
        if self.__states[next_state()] == self.__state_types.Goal:
            reward = 1  # TODO: define an appropriate reward value
            flag = True
        if self.__states[next_state()] == self.__state_types.Obstacle:
            reward = -10
        else:
            pass  # default

        # commit
        self.s = copy.deepcopy(next_state)  # update current state

        return self.flatten_index(next_state), reward, flag, info


# TODO: generalization
def create_world(
        n_size: int,
        state_type: State_type = State_type(['Normal', 'Start', 'Goal', 'Obstacle']),
        start: Cord = Cord(0, 0),
        goals: list = [],
        obstacles: list = []):
    # initialization
    if len(goals) == 0:
        goals = [Cord(n_size - 1, n_size - 1)]
    world = np.full((n_size, n_size), state_type.Normal)

    # set Start point
    world[start()] = state_type.Start

    # set Goal points
    for goal in goals:
        if world[goal()] != state_type.Normal:
            raise Exception(
                'The cordinate of Goal cannot be the same as Start or Obstacles: ',
                goal
            )
        world[goal()] = state_type.Goal

    # set Obstacle points
    for obstacle in obstacles:
        if world[obstacle()] != state_type.Normal:
            raise Exception(
                'The cordinate of Obstacle cannot be the same as Start or Goals: ',
                obstacle
            )
        world[obstacle()] = state_type.Obstacle

    return world


# TODO: argparse
if __name__ == "__main__":
    # set state_type and action
    state_types = State_type(['Normal', 'Start', 'Goal', 'Obstacle'])
    actions = Action(['Up', 'Down', 'Left', 'Right'])
    # pprint(state_type)
    # pprint(action)

    # create world(state) and env.
    size = 5
    goals = [Cord(3, 3)]  # starting from 0
    obstacles = [Cord(2, 3), Cord(1, 2), Cord(3, 2), Cord(3, 4)]
    states = create_world(size, goals=goals, obstacles=obstacles)
    # pprint(states)
    env = Env(state_types, states, actions)

    # create Q-Table and initialize with 0
    Q = np.zeros([states.size, actions.size])
    # pprint(Q)

    # set learning parameters
    lr = 0.8  # learning rate
    y = 0.95  # discount factor
    episodes_num = 2000
    x = 0.998  # 0.998 ** 2000 == 0.018242425223750636  # TODO: auto. setting using episode_num

    Gs = []

    for i in range(episodes_num):
        s = env.reset()  # Reset environment and get first new observation

        G = 0  # total reward
        d = False  # precess state (end) flag
        step_num = 100

        for j in range(step_num):
            print('(Episode: %5d, Steps: %5d)' % (i, j), end='\r')

            # Q[s, :] returns probability of every possible actions in current state
            q = Q[s, :]

            # choose an action e-greedly with Q-table
            if random.random() < x:
                # Exploration method called 'decaying'
                # choose an action randomly (epsilon)
                a = random.randint(0, actions.size - 1)
            else:
                # np.argmax returns an index which has the maximun value into the flattened array
                # np.unravel_index returns the position (row and col) from original array. For example:
                # np.unravel_index(8, np.array(L).shape)  # where 8 is return value of argmax, L is the original array
                # np.random.randn() adds gaussian random noise on Q-table for extra exploration
                # Exploration method called 'random noise' also using decaying
                q += np.random.randn(actions.size, ) * (1.0 / (i + 1))
                # pprint(q)

                # Do not use bare np.argmax because it can not treat the case of existing multiple max values
                # But we add random noise on q, so it is fine to use argmax because there are really few chance to get same value
                a = random.choice(np.where(q == q.max())[0])
                # pprint(a)

            # get new state, reward, flag, and extra info.
            s1, r, d, info = env.step(a)
            # print(actions[a], s1, r, d)

            """
            # update Q-Table with new knowledge(=reward)
            # Q(s, a) = r + y * max(Q(s', a') for all a')
            # it seems like SGD... if 'a' is good action: Q[s, a]++ else Q[s, a]--
            """
            Q[s, a] += lr * (r + y * np.max(Q[s1, :]) - Q[s, a])

            G += r
            s = s1
            if d:
                # if G >= 1:
                #     print('(Episode: %5d, Steps: %5d) We arrive at goal state.' % (i, j))
                break

            # pprint(Q)

        """
        # history
        # current implementation ignores obstacle's panalty
        # TODO: (WIP) the less step, the better score
        """
        # Gs.append(G)
        Gs.append(1 if G >= 1 else 0)

    # visualization
    print()  # clear
    print('Score over time:', (sum(Gs) / episodes_num))
    plt.title('Score over time')
    plt.plot(Gs)
    plt.show()
    plt.close()

    # show Q-Table
    print('Final Q-Table:')

    # normalization
    for q in Q:
        if sum(q) != 0.0:
            q /= sum(q)

    pprint(np.round(Q, 3))  # rounds

    # show map
    print('Map:')
    world_map = [[] for _ in range(size)]
    for i, row_elems in enumerate(states):
        for elem in row_elems:
            # using str.ljust for pedding
            world_map[i].append(state_types[elem].ljust(9, ' '))

    Q_map = np.argmax(Q, axis=1).reshape(
        size, size)  # get argmax from row-by-row
    Q_map_str = [[] for _ in range(size)]
    for i, row_elems in enumerate(Q_map):
        for elem in row_elems:
            # using str.ljust for pedding
            Q_map_str[i].append(actions[elem].ljust(9, ' '))

    pprint(world_map)
    pprint(Q_map_str)
