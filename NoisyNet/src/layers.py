import keras.backend as K
from keras import initializers, activations
from keras.engine.topology import Layer


# TODO: variables convert into private
# Buile own layer
# Ref: https://keras.io/ko/layers/writing-your-own-keras-layers/
class NoisyDense(Layer):
    def __init__(self,
                 output_dim,
                 activation=None,
                 **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)

        self.e_i = None
        self.e_j = None

        super(NoisyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert isinstance(input_shape, list)

        self.input_dim = input_shape[-1]

        # Factorised NoisyNet
        # Ref: https://github.com/jakegrigsby/keras-rl/blob/master/rl/layers.py
        sqrt_inputs = self.input_dim ** (1 / 2)
        self.sigma_initializer = initializers.Constant(value=0.5 / sqrt_inputs)
        self.mu_initializer = initializers.RandomUniform(minval=(-1.0 / sqrt_inputs), maxval=(1.0 / sqrt_inputs))

        # Learnable parameters
        # TODO: constraint, regularizer
        self.mu_weight = self.add_weight(name='mu_weights',
                                         shape=(self.input_dim, self.output_dim),
                                         initializer=self.mu_initializer,
                                         trainable=True)

        self.sigma_weight = self.add_weight(name='sigma_weights',
                                            shape=(self.input_dim, self.output_dim),
                                            initializer=self.sigma_initializer,
                                            trainable=True)

        self.mu_bias = self.add_weight(name='mu_bias',
                                       shape=(self.output_dim,),
                                       initializer=self.mu_initializer,
                                       trainable=True)

        self.sigma_bias = self.add_weight(name='sigma_bias',
                                          shape=(self.output_dim,),
                                          initializer=self.sigma_initializer,
                                          trainable=True)

        self.reset_noise()
        super(NoisyDense, self).build(input_shape)

    def call(self, x):
        # assert isinstance(x, list)

        # Factorised Gaussian noise
        def f(e):
            return K.sign(e) * (K.sqrt(K.abs(e)))

        eW = f(self.e_i) * f(self.e_j)
        eB = f(self.e_j)

        noise_injected_weights = self.mu_weight + (self.sigma_weight * eW)
        noise_injected_bias = self.mu_bias + (self.sigma_bias * eB)
        output = K.bias_add(K.dot(x, noise_injected_weights), noise_injected_bias)

        if self.activation != None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)

        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    # Ref: https://github.com/LuEE-C/Noisy-A3C-Keras
    def reset_noise(self):
        # Random variables
        # sample from noise distribution
        self.e_i = K.random_normal((self.input_dim, self.output_dim))
        self.e_j = K.random_normal((self.output_dim,))
