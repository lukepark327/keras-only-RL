import keras.backend as K
from keras.layers import Dense, Input, Lambda, Multiply
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import random
import math

from memories import ReplayMemory


class Agent:
    def __init__(self, n_actions, world_size, lr, y, e, num_atoms):
        self.n_actions = n_actions
        self.world_size = world_size
        self.size = int(np.sqrt(self.world_size))

        self.lr = lr
        self.y = y
        self._e0 = e
        self.e = self._e0
        self._noise = 1.0
        self.noise = self._noise

        n_inputs = self.world_size
        n_outputs = self.n_actions

        # C51
        # TODO: argparse
        self.num_atoms = num_atoms  # assert(num_atoms > 1)
        self.v_max = 1  # Max possible score
        self.v_min = -100  # Min possible score
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        self.Z, self.Q = self._build_model(n_inputs, n_outputs)

        # Use categorical_crossentropy
        self.Z.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))

        # Fixed Q-target
        self.Z_target, _ = self._build_model(n_inputs, n_outputs)
        self.update_target()

        # TODO: save_weights
        # TODO: load_weights

        # Replay Memory
        # TODO: argparse
        self.batch_size = 64
        self.memory = ReplayMemory(2000)
        self.train_start = 1000

    def _build_model(self, n_inputs, n_outputs, num_atoms=51):
        x = Input(shape=(n_inputs, ), name='state')
        f = Dense(24, activation='relu')(x)
        f = Dense(24, activation='relu')(f)
        f = Dense(24, activation='relu')(f)

        # Z
        # C51: num_atoms is 51
        Ps = []
        for i in range(n_outputs):
            Ps.append(
                Dense(num_atoms, activation='softmax', name='action_%s' % str(i))(f)
            )
        Z_model = Model(inputs=x, outputs=Ps)

        # Q
        # convert list of tensors into high-dim tensor: (?, 4, 51)
        P = Lambda(
            lambda p: K.stack(p[:], axis=1)
        )(Ps)
        Q = Lambda(
            lambda p: K.sum(p[:, :, :] * self.z, axis=-1)
        )(P)
        Q_model = Model(inputs=x, outputs=Q)

        # Z_model.summary()
        # Q_model.summary()
        return Z_model, Q_model

    def _one_hot_encoded(self, cord):
        encoded = np.zeros(self.world_size)
        encoded[self._flatten(cord)] = 1.0
        return encoded.reshape([1, self.world_size])

    def _flatten(self, cord):
        return cord[0] * self.size + cord[1]

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

        experiences = self.memory.sample(batch_size)
        inputs = []
        outputs = [np.zeros((batch_size, self.num_atoms)) for t in range(self.n_actions)]  # (4, ?, 51)

        for i, exp in enumerate(experiences):
            state, action, reward, next_state, done = exp

            # TODO: optimize
            z_values = self.Z.predict(self._one_hot_encoded(state))  # list of np.array: (4, 1, 51)
            # Preserving the other values except z_values[action]
            for t in range(self.n_actions):
                if t == action:
                    pass
                else:
                    for j in range(self.num_atoms):
                        outputs[t][i][j] += z_values[t][0][j]

            # Ref: https://github.com/flyyufelix/C51-DDQN-Keras
            if done:
                Tz = min(self.v_max, max(self.v_min, reward))  # only reward
                bj = (Tz - self.v_min) / self.delta_z  # index
                m_l, m_u = math.floor(bj), math.ceil(bj)
                outputs[action][i][int(m_l)] += (m_u - bj)  # (m_u - bj) * 1
                outputs[action][i][int(m_u)] += (bj - m_l)  # (bj - m_l) * 1
            else:
                # Like DDQN
                q_values_next = self.Q.predict(self._one_hot_encoded(next_state))[0]
                selected_action = np.argmax(q_values_next)
                estimated_values = self.Z_target.predict(self._one_hot_encoded(next_state))[selected_action][0]

                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, reward + self.y * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z  # index
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    outputs[action][i][int(m_l)] += (m_u - bj) * estimated_values[j]
                    outputs[action][i][int(m_u)] += (bj - m_l) * estimated_values[j]

            inputs.append(self._one_hot_encoded(state)[0])

        self.Z.fit(
            np.array(inputs),
            outputs,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

    def update_target(self):
        self.Z_target.set_weights(self.Z.get_weights())
