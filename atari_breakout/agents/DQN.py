from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np
import random

from agents.memories import ReplayMemory
from agents.losses import Huber_loss


class DQNAgent:
    def __init__(self, action_size):
        self.render = True
        self.load_model_dir = False

        # Environment settings
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        # Epsilon
        self.epsilon = 1.  # Current e
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.exploration_steps

        # Training
        self.no_op_steps = 30
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = ReplayMemory(400000)

        # Build model
        self.optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        self.model_for_train, self.model = self._build_model()
        _, self.target_model = self._build_model()
        self.update_target_model()
        if self.load_model_dir:
            # TODO: mkdir
            self.load_model("./save_model/breakout.h5")

        # for logging
        # TODO: Tensorboard
        self.avg_q_max, self.avg_loss = 0.0, 0.0

    def _build_model(self):
        x = Input(shape=self.state_size, name='input')
        h = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(x)
        h = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(h)
        h = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(h)
        h = Flatten()(h)
        h = Dense(512, activation='relu')(h)
        y_pred = Dense(self.action_size, activation='linear', name='y_pred')(h)

        # for custom loss function
        y_true = Input(shape=(self.action_size, ), name='y_true')
        model_for_train = Model(
            inputs=[x, y_true],
            outputs=y_pred,
            name='model_for_train'
        )
        model_for_train.add_loss(Huber_loss(y_true, y_pred))
        model_for_train.compile(loss=None, optimizer=self.optimizer)

        model_for_using = Model(
            inputs=x,
            outputs=y_pred,
            name='model_for_using'
        )

        model_for_using.summary()
        return model_for_train, model_for_using

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state / 255.)
            q_value = self.model.predict(state)[0]
            return np.argmax(q_value)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def _decaying(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay

    def learn(self):
        if len(self.memory) < self.train_start:
            return

        self._decaying()

        batch_size = min(len(self.memory), self.batch_size)

        states = np.zeros((batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_states = np.zeros((batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        actions, rewards, dones = [], [], []  # Actually, not done but dead.

        experiences = self.memory.sample(batch_size)
        for i, experience in enumerate(experiences):
            # 'State', 'Action', 'Reward', 'Next_state', 'Done'
            states[i] = np.float32(experience[0] / 255.)
            actions.append(experience[1])
            rewards.append(experience[2])
            next_states[i] = np.float32(experience[3] / 255.)
            dones.append(experience[4])

        target_values = self.target_model.predict(next_states)
        targets = np.zeros((batch_size, self.action_size, ))
        for i in range(batch_size):
            targets[i] = target_values[i]
            action = actions[i]
            if dones[i]:
                targets[i][action] = rewards[i]
            else:
                targets[i][action] = rewards[i] + self.discount_factor * np.amax(target_values[i])

        metrics = self.model_for_train.fit(
            [states, targets],
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

        self.avg_loss += metrics.history['loss'][0]

    def save_model(self, _dir):
        self.model.save_weights(_dir)

    def load_model(self, _dir):
        self.model.load_weights(_dir)
