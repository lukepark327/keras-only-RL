from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import random

import arguments


# arguments
args = arguments.parser()
lr = args.lr  # learning rate
y = args.y  # discount factor
e = args.e  # e-greedy factor
n_episodes = args.r  # episode number (round)
n_steps = args.s  # step number
print("> Setting:", args)


class Agent:
    def __init__(self, n_actions, world_size):
        self.n_actions = n_actions
        self.world_size = world_size
        self.size = int(np.sqrt(world_size))

        self.lr = lr
        self.y = y
        self._e0 = e
        self.e = self._e0
        self._noise = 1.0
        self.noise = self._noise

        n_inputs = world_size
        n_outputs = self.n_actions
        self.Q_NN = self._build_model(n_inputs, n_outputs)
        self.Q_NN.compile(loss='mse', optimizer=Adam(lr=self.lr))
        # TODO: save_weights
        # TODO: load_weights

    def _build_model(self, n_inputs, n_outputs):
        inputs = Input(shape=(n_inputs, ), name='state')
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(n_outputs, activation='linear', name='action')(x)
        Q_model = Model(inputs, x)
        # Q_model.summary()
        return Q_model

    def _flatten(self, cord):
        return cord[0] * self.size + cord[1]

    def _one_hot_encoded(self, cord):
        encoded = np.zeros(self.world_size)
        encoded[self._flatten(cord)] = 1.0
        return encoded.reshape([1, self.world_size])

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
            q = self.Q_NN.predict(self._one_hot_encoded(state))[0]

            # Exploration method called 'random noise' also using decaying
            # np.random.randn() adds gaussian random noise on Q-table for extra exploration
            q += np.random.randn(self.n_actions, ) * self.noise

            # Do not use bare np.argmax because it can not treat the case of existing multiple max values
            # But we add random noise on q, so it is fine to use argmax because there are really few chance to get same value
            action = random.choice(np.where(q == q.max())[0])

        return action

    def learn(self, state, action, reward, next_state):
        q_values = self.Q_NN.predict(self._one_hot_encoded(state))[0]
        q2 = reward + self.y * np.max(self.Q_NN.predict(self._one_hot_encoded(next_state))[0])

        q_values[action] = q2

        self.Q_NN.fit(
            np.array(self._one_hot_encoded(state)),
            np.array([q_values]),
            epochs=1,
            verbose=0
        )

    def get_Q_table(self):
        Q_table = np.zeros([self.world_size, self.n_actions])
        for r, _ in enumerate(Q_table):
            encoded = np.zeros(self.world_size)
            encoded[r] = 1.0
            encoded = encoded.reshape([1, self.world_size])
            Q_table[r] = self.Q_NN.predict(encoded)[0]
        return Q_table


if __name__ == "__main__":
    from pprint import pprint

    from visualization import show_list, map_print
    from env import Env

    n_agents = 1
    env = Env(n_agents)
    agents = [Agent(env.action_space.size, env.world.size) for _ in range(n_agents)]

    Gs = []  # Save total reward for each episode  # TODO: multiple agents
    for i in range(n_episodes):
        # initialization
        states = env.reset()  # list of agent_states
        G = 0  # total reward  # TODO: multiple agents

        for n, agent in enumerate(agents):
            for j in range(n_steps):
                print('(Episode: %5d, Agent: %5d, Steps: %5d)' % (i, n, j), end='\r')

                action = agent.get_action(i, states[n])
                next_state, reward, done, _ = env.step(n, action)
                agent.learn(states[n], action, reward, next_state)

                G += reward  # total reward
                states[n] = next_state  # The agent goes to the next state

                if done:
                    break

        Gs.append(G)  # TODO: multiple agents

    # visualization
    # TODO: tensorboard
    print()  # clear
    print('Score over time:', (sum(Gs) / n_episodes))
    show_list('Score over time', Gs)

    # show Q-Table
    print('Final Q-Table driven by NN:')
    pprint(np.round(agents[0].get_Q_table(), 3))  # rounds

    # show map
    print('Map:')
    world_map = map_print(env.size, env.world, env.state_space)
    pprint(world_map)

    print('Q-map:')
    Q_map = np.argmax(agents[0].get_Q_table(), axis=1).reshape(env.size, env.size)  # get argmax from row-by-row
    Q_map_str = map_print(env.size, Q_map, env.action_space)
    pprint(Q_map_str)
