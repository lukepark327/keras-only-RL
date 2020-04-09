import keras.backend as K
from keras import initializers, activations
from keras.engine.topology import Layer


# Buile own layer
# Ref: https://keras.io/ko/layers/writing-your-own-keras-layers/
class NoisyDense(Layer):
    def __init__(self,
                 output_dim,
                 activation=None,
                 **kwargs):
        super(NoisyDense, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.activation = activations.get(activation)

    def build(self, input_shape):
        # assert isinstance(input_shape, list)

        self.input_dim = input_shape[-1]

        # Factorised NoisyNet
        # Ref: https://github.com/LuEE-C/Noisy-A3C-Keras/blob/ba121a296c644c2b634b485f21769d7d5667fbad/NoisyDense.py#L91
        # Ref: https://github.com/jakegrigsby/keras-rl/blob/b4bef96a36e12f8e1292bd0e5c63b6a4663466eb/rl/layers.py
        sqrt_inputs = self.input_dim ** (1 / 2)
        self.sigma_initializer = initializers.Constant(value=0.5 / sqrt_inputs)
        self.mu_initializer = initializers.RandomUniform(minval=(-1.0 / sqrt_inputs), maxval=(1.0 / sqrt_inputs))

        # Learnable parameters
        # TODO: constraint, regularizer
        self.mu_weight = self.add_weight(name='mu_weights',
                                         shape=(self.input_dim, self.output_dim),
                                         initializer=self.mu_initializer)

        self.sigma_weight = self.add_weight(name='sigma_weights',
                                            shape=(self.input_dim, self.output_dim),
                                            initializer=self.sigma_initializer)

        self.mu_bias = self.add_weight(name='mu_bias',
                                       shape=(self.output_dim,),
                                       initializer=self.mu_initializer)

        self.sigma_bias = self.add_weight(name='sigma_bias',
                                          shape=(self.output_dim,),
                                          initializer=self.sigma_initializer)

        super(NoisyDense, self).build(input_shape)

    def call(self, x):
        # assert isinstance(x, list)

        # Random variables
        # sample from noise distribution
        e_i = K.random_normal((self.input_dim, self.output_dim))
        e_j = K.random_normal((self.output_dim,))

        # Factorised Gaussian noise
        def f(e):
            return K.sign(e) * (K.sqrt(K.abs(e)))

        eW = f(e_i) * f(e_j)
        eB = f(e_j)

        noise_injected_weights = K.dot(x, self.mu_weight + (self.sigma_weight * eW))
        noise_injected_bias = self.mu_bias + (self.sigma_bias * eB)
        output = K.bias_add(noise_injected_weights, noise_injected_bias)

        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)

        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    # TODO: remove_noise()
    # TODO: Is this function needed?
