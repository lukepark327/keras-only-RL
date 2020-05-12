import numpy as np

from agents.DQN import DQNAgent


class DDQNAgent(DQNAgent):
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

        values = self.model.predict(next_states)  # DDQN
        target_values = self.target_model.predict(next_states)
        targets = np.zeros((batch_size, self.action_size, ))
        for i in range(batch_size):
            targets[i] = target_values[i]
            action = actions[i]
            if dones[i]:
                targets[i][action] = rewards[i]
            else:
                selected_action = np.argmax(values[i])
                targets[i][action] = rewards[i] + self.discount_factor * targets[i][selected_action]

        metrics = self.model_for_train.fit(
            [states, targets],
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

        self.avg_loss += metrics.history['loss'][0]
