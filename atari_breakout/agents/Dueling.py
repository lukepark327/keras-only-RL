import keras.backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Add
from keras.models import Model

from agents.losses import Huber_loss

from agents.DQN import DQNAgent
from agents.Double import DDQNAgent


class DuelingAgent(DQNAgent):
    def _build_model(self):
        x = Input(shape=self.state_size, name='input')
        h = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(x)
        h = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(h)
        h = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(h)
        flatten = Flatten()(h)

        """
        # Directly summing V and A gives us no guarantees that the A will actually predict the V.
        # Instead we combine them with criterion - Max or Avg
        # This(Avg) loses the original semantics of V and A (c.f. Max) .
        # But on the other hand it increases the stability of the optimization.
        # Ref: Dueling Network Architectures for Deep Reinforcement Learning
        """
        # Advantage function
        A = Dense(512, activation='relu')(flatten)
        A = Dense(self.action_size, activation='linear', name='advanced')(A)
        A = Lambda(
            lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
            output_shape=(self.action_size, )
        )(A)

        # Value function
        V = Dense(512, activation='relu')(flatten)
        V = Dense(1, activation='linear', name='value')(V)
        V = Lambda(
            lambda v: K.expand_dims(v[:, 0], -1),
            output_shape=(self.action_size, )
        )(V)

        y_pred = Add()([V, A])  # tensor shape broadcasting

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


class D3QNAgent(DDQNAgent):
    def _build_model(self):
        x = Input(shape=self.state_size, name='input')
        h = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(x)
        h = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(h)
        h = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(h)
        flatten = Flatten()(h)

        """
        # Directly summing V and A gives us no guarantees that the A will actually predict the V.
        # Instead we combine them with criterion - Max or Avg
        # This(Avg) loses the original semantics of V and A (c.f. Max) .
        # But on the other hand it increases the stability of the optimization.
        # Ref: Dueling Network Architectures for Deep Reinforcement Learning
        """
        # Advantage function
        A = Dense(512, activation='relu')(flatten)
        A = Dense(self.action_size, activation='linear', name='advanced')(A)
        A = Lambda(
            lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
            output_shape=(self.action_size, )
        )(A)

        # Value function
        V = Dense(512, activation='relu')(flatten)
        V = Dense(1, activation='linear', name='value')(V)
        V = Lambda(
            lambda v: K.expand_dims(v[:, 0], -1),
            output_shape=(self.action_size, )
        )(V)

        y_pred = Add()([V, A])  # tensor shape broadcasting

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
