import numpy as np
from matplotlib import pyplot as plt


def scattering(title: str, L: list):
    # TODO: pretty
    plt.title(title)
    plt.scatter(range(len(L)), L)
    plt.show()
    plt.close()


def map_print(size, matrix, names):
    map = [[] for _ in range(size)]
    for i, row_elems in enumerate(matrix):
        for elem in row_elems:
            map[i].append(names._fields[elem].ljust(9, ' '))  # using str.ljust for pedding
    return map


def NN_to_Table(NN, world_size, n_actions):
    Table = np.zeros([world_size, n_actions])

    for r, _ in enumerate(Table):
        encoded = np.zeros(world_size)
        encoded[r] = 1.0
        encoded = encoded.reshape([1, world_size])
        Table[r] = NN.predict(encoded)[0]

    return Table
