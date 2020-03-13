import numpy as np
from pprint import pprint

import arguments
from utils import Cord, string_to_Cord
from env import State_type, Env
from agent import Action, Agent
from visual import show_list, map_print


def create_world(
        n_size: int,
        state_type: State_type,
        start: Cord,
        goals: set,
        obstacles: set):

    # initialization
    if len(goals) == 0:
        goals = {Cord(n_size - 1, n_size - 1)}

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


def step(env: Env, agent: Agent, a: int):
    # initialization
    next_state = Cord(0, 0)
    reward = 0
    flag = False
    info = []  # extra info.

    # get next state
    if a == agent.actions.Up:
        next_state = Cord(max(env.s.x - 1, 0), env.s.y)
    elif a == agent.actions.Down:
        next_state = Cord(min(env.s.x + 1, env.size - 1), env.s.y)
    elif a == agent.actions.Left:
        next_state = Cord(env.s.x, max(env.s.y - 1, 0))
    elif a == agent.actions.Right:
        next_state = Cord(env.s.x, min(env.s.y + 1, env.size - 1))
    else:
        pass  # default

    # get reward and flag
    if env.states[env.s()] == env.state_types.Goal:
        reward = 1
        flag = True
    if env.states[env.s()] == env.state_types.Obstacle:
        reward = -10
    else:
        pass  # default

    # commit
    env.s = next_state  # update current state

    return next_state.flatten_index(env.size), reward, flag, info


if __name__ == "__main__":

    """arguments"""
    args = arguments.parser()
    lr = args.lr  # learning rate
    y = args.y  # discount factor
    episodes_num = args.r  # episode number (round)
    x = args.x  # e-greedy factor
    step_num = args.s  # step number
    size = args.size  # map size (N x N)
    start = string_to_Cord(args.start)  # a start cordinate
    goals = string_to_Cord(args.goals)  # set of goal cordinate(s)
    obstacles = string_to_Cord(args.obs)  # set of obstacle cordinate(s)
    print("> Setting:", args)

    """set action and agent"""
    actions = Action(['Up', 'Down', 'Left', 'Right'])
    agent = Agent(actions, x)  # one agent

    """create world(state) and env."""
    state_types = State_type(['Normal', 'Start', 'Goal', 'Obstacle'])
    states = create_world(size, state_types, start, goals, obstacles)
    # pprint(states)
    env = Env(state_types, states, Cord(0, 0))

    """create Q-Table and initialize with 0"""
    Q = np.zeros([states.size, actions.size])
    # pprint(Q)

    """rounds"""
    Gs = []  # Save total reward for each episode
    for i in range(episodes_num):

        # initialization
        s = env.reset()  # Reset environment and get first new observation
        G = 0  # total reward
        d = False  # precess state (end) flag

        for j in range(step_num):
            print('(Episode: %5d, Steps: %5d)' % (i, j), end='\r')
            q = Q[s, :]  # Q[s, :] returns probability of every possible actions in current state
            a = agent.do(q, i)  # q is updated because of call-by-reference
            s1, r, d, info = step(env, agent, a)  # get new state, reward, flag, and extra info.
            # print(actions[a], s1, r, d, info)

            """
            # update Q-Table with new knowledge(=reward)
            # Q(s, a) = r + y * max(Q(s', a') for all a')
            """
            Q[s, a] += lr * (r + y * np.max(Q[s1, :]) - Q[s, a])

            G += r  # total reward
            s = s1  # An agent goes to the next state
            if d:
                break

        """history"""
        # Gs.append(G)
        Gs.append(1 if G >= 1 else 0)

        """for next episode"""
        agent.x_update()

    """visualization"""
    print()  # clear
    print('Score over time:', (sum(Gs) / episodes_num))
    show_list('Score over time', Gs)

    # show Q-Table
    print('Final Q-Table:')
    pprint(np.round(Q, 3))  # rounds

    # show map
    print('Map:')
    world_map = map_print(size, states, state_types)
    pprint(world_map)

    print('Q-map:')
    Q_map = np.argmax(Q, axis=1).reshape(size, size)  # get argmax from row-by-row
    Q_map_str = map_print(size, Q_map, actions)
    pprint(Q_map_str)
