import numpy as np


def NN_to_Table(NN, world_size, n_actions):
    Table = np.zeros([world_size, n_actions])

    for r, _ in enumerate(Table):
        encoded = np.zeros(world_size)
        encoded[r] = 1.0
        encoded = encoded.reshape([1, world_size])
        Table[r] = NN.predict(encoded)[0]

    return Table
