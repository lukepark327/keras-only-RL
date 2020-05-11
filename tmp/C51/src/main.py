import numpy as np
from pprint import pprint

import arguments
from env import Env
from agent import Agent
from visualization import scattering, map_print, NN_to_Table


if __name__ == "__main__":
    # arguments
    args = arguments.parser()
    lr = args.lr  # learning rate
    y = args.y  # discount factor
    e = args.e  # e-greedy factor
    n_episodes = args.r  # episode number (round)
    n_steps = args.s  # step number
    print("> Setting:", args)

    update_target_interval = 10  # per episode  # TODO: argparse

    env = Env()
    agent = Agent(len(env.actions), env.world.size, lr, y, e, 51)  # TODO: argparse

    Gs = []  # total rewards for each episode

    for i in range(n_episodes):
        # initialization
        state = env.reset()  # agent's initial state
        G = 0  # total reward

        for j in range(n_steps):
            print('(Episode: %5d, Steps: %5d)' % (i, j), end='\r')

            action = agent.get_action(i, state)
            next_state, reward, done, _ = env.step(action)
            agent.append_sample(state, action, reward, next_state, done)
            agent.learn()

            G += reward  # total reward
            state = next_state  # The agent goes to the next state

            if done:
                break

        Gs.append(G)

        # update Q_target
        if i % update_target_interval == 0:
            agent.update_target()

    # visualization
    # TODO: tensorboard
    print()  # clear
    print('Score over time:', (sum(Gs) / n_episodes))
    scattering('Score over time', Gs)

    # show Q-Table
    Q_table = NN_to_Table(agent.Q, agent.world_size, agent.n_actions)

    print('Final Q-Table:')
    pprint(np.round(Q_table, 3))  # rounds

    # show map
    print('Map:')
    world_map = map_print(env.side_size, env.world, env.states)
    pprint(world_map)

    print('Q-map:')
    Q_map = np.argmax(Q_table, axis=1).reshape(env.side_size, env.side_size)  # get argmax from row-by-row
    Q_map_str = map_print(env.side_size, Q_map, env.actions)
    pprint(Q_map_str)
