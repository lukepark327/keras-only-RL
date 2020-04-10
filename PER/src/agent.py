from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import random

from memories import PrioritizedReplayMemory
from losses import PER_loss


class Agent:
    def __init__(self, n_actions, world_size, lr, y, e, r):
        self.n_actions = n_actions
        self.world_size = world_size
        self.size = int(np.sqrt(self.world_size))

        self.lr = lr
        self.y = y
        self._e0 = e
        self.e = self._e0
        self.n_episodes = r
        self._noise = 1.0
        self.noise = self._noise

        n_inputs = self.world_size
        n_outputs = self.n_actions
        self.Q_train, self.Q = self._build_model(n_inputs, n_outputs, trainable=True)

        # Fixed Q-target
        _, self.Q_target = self._build_model(n_inputs, n_outputs, trainable=False)
        self.update_target()

        # TODO: save_weights
        # TODO: load_weights

        # Prioritized Replay Memory (PER)
        # TODO: argparse
        self.alpha = 0.6
        self._beta0 = 0.4
        self.beta = self._beta0
        self.epsilon = 10 ** -6

        self.batch_size = 64
        self.memory = PrioritizedReplayMemory(2000)
        self.train_start = 1000

    def _build_model(self, n_inputs, n_outputs, trainable=False):
        x = Input(shape=(n_inputs, ), name='in')
        f = Dense(24, activation='relu')(x)
        f = Dense(24, activation='relu')(f)
        f = Dense(24, activation='relu')(f)
        y_pred = Dense(n_outputs, activation='linear', name='y_pred')(f)

        # for custom fit
        y_true = Input(shape=(n_outputs, ), name='y_true')
        importances = Input(shape=(1, ), name='importances')

        Q_train_model = Model(
            inputs=[x, y_true, importances],
            outputs=y_pred,
            name='train_only'
        )

        if trainable:
            # custom loss function
            Q_train_model.add_loss(PER_loss(y_true, y_pred, importances))
            Q_train_model.compile(loss=None, optimizer=Adam(lr=self.lr))
        else:
            pass

        # testing model for easier use.
        Q_model = Model(inputs=x, outputs=y_pred, name='test_only')

        # Q_model.summary()
        return Q_train_model, Q_model

    def _one_hot_encoded(self, cord):
        encoded = np.zeros(self.world_size)
        encoded[self._flatten(cord)] = 1.0
        return encoded.reshape([1, self.world_size])

    def _flatten(self, cord):
        return cord[0] * self.size + cord[1]

    def _decaying(self, episode_num):
        self.e = self._e0 ** episode_num
        self.noise = self._noise / (episode_num + 1)

        self.beta = self._beta0 + (episode_num / self.n_episodes) * (1.0 - self._beta0)
        self.beta = min(self.beta, 1.0)

    def get_action(self, episode_num, state):
        self._decaying(episode_num)

        # choose an action e-greedly
        if random.random() < self.e:
            # Exploration method called 'decaying'
            # choose an action randomly (epsilon)
            action = random.randint(0, self.n_actions - 1)
        else:
            q = self.Q.predict(self._one_hot_encoded(state))[0]

            # Exploration method called 'random noise' also using decaying
            # np.random.randn() adds gaussian random noise on Q-table for extra exploration
            q += np.random.randn(self.n_actions, ) * self.noise

            # Do not use bare np.argmax because it can not treat the case of existing multiple max values
            # But we add random noise on q, so it is fine to use argmax because there are really few chance to get same value
            action = random.choice(np.where(q == q.max())[0])

        return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(len(self.memory), self.batch_size)

        indices, experiences, importances = self.memory.sample(batch_size, self.beta)
        inputs = []
        outputs = []
        for idx, exp in zip(indices, experiences):
            state, action, reward, next_state, done = exp

            q_values = self.Q.predict(self._one_hot_encoded(state))[0]

            # TD-error
            if done:
                q2 = reward
            else:
                q2 = reward + self.y * np.max(self.Q_target.predict(self._one_hot_encoded(next_state))[0])

            q1 = q_values[action]
            q_values[action] = q2

            # Ref: https://github.com/Parsa33033/Deep-Reinforcement-Learning-DQN/blob/master/DQN-with-Prioritized-Experience-Replay.py
            self.memory.update_priority(
                idx,
                (np.abs(q2 - q1) + self.epsilon) ** self.alpha  # proportional variant
            )

            inputs.append(self._one_hot_encoded(state)[0])
            outputs.append(q_values)

        self.Q_train.fit(
            [np.array(inputs), np.array(outputs), importances],
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

    def update_target(self):
        self.Q_target.set_weights(self.Q.get_weights())
