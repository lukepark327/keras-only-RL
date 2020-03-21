from matplotlib import pyplot as plt

from utils import Enum


def show_list(title: str, L: list):
    # TODO: pretty
    plt.title(title)
    plt.scatter(range(len(L)), L)
    plt.show()
    plt.close()


def map_print(size, matrix, enum: Enum):
    map = [[] for _ in range(size)]
    for i, row_elems in enumerate(matrix):
        for elem in row_elems:
            map[i].append(enum[elem].ljust(9, ' '))  # using str.ljust for pedding
    return map
