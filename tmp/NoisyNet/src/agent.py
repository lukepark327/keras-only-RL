from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import random

from memories import ReplayMemory
from layers import NoisyDense


class Agent:
    def __init__(self, n_actions, world_size, lr, y):
        self.n_actions = n_actions
        self.world_size = world_size
        self.size = int(np.sqrt(self.world_size))

        self.lr = lr
        self.y = y

        n_inputs = self.world_size
        n_outputs = self.n_actions
        self.Q = self._build_model(n_inputs, n_outputs)
        self.Q.compile(loss='mse', optimizer=Adam(lr=self.lr))

        # Fixed Q-target
        self.Q_target = self._build_model(n_inputs, n_outputs)
        self.update_target()

        # TODO: save_weights
        # TODO: load_weights

        # Replay Memory
        # TODO: argparse
        self.batch_size = 64
        self.memory = ReplayMemory(2000)
        self.train_start = 1000

    def _build_model(self, n_inputs, n_outputs):
        inputs = Input(shape=(n_inputs, ), name='state')
        x = NoisyDense(24, activation='relu', name='h1')(inputs)
        x = NoisyDense(24, activation='relu', name='h2')(x)
        x = NoisyDense(24, activation='relu', name='h3')(x)
        x = NoisyDense(n_outputs, activation='linear', name='action')(x)
        Q_model = Model(inputs, x)
        # Q_model.summary()
        return Q_model

    def _reset_noise(self, NN):
        NN.get_layer('h1').reset_noise()
        NN.get_layer('h2').reset_noise()
        NN.get_layer('h3').reset_noise()
        NN.get_layer('action').reset_noise()

    def _one_hot_encoded(self, cord):
        encoded = np.zeros(self.world_size)
        encoded[self._flatten(cord)] = 1.0
        return encoded.reshape([1, self.world_size])

    def _flatten(self, cord):
        return cord[0] * self.size + cord[1]

    def get_action(self, episode_num, state):
        q = self.Q.predict(self._one_hot_encoded(state))[0]
        action = random.choice(np.where(q == q.max())[0])
        return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(len(self.memory), self.batch_size)

        experiences = self.memory.sample(batch_size)
        inputs = []
        outputs = []
        for exp in experiences:
            state, action, reward, next_state, done = exp

            q_values = self.Q.predict(self._one_hot_encoded(state))[0]

            if done:
                q2 = reward
            else:
                q2 = reward + self.y * np.max(self.Q_target.predict(self._one_hot_encoded(next_state))[0])

            q_values[action] = q2

            inputs.append(self._one_hot_encoded(state)[0])
            outputs.append(q_values)

        self.Q.fit(
            np.array(inputs),
            np.array(outputs),
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

        self._reset_noise(self.Q)
        self._reset_noise(self.Q_target)

    def update_target(self):
        self.Q_target.set_weights(self.Q.get_weights())
