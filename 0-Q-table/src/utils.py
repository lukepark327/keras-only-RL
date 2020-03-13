from ast import literal_eval


class Enum:
    def __init__(self, items: list):
        self.__items = items
        self.size = len(items)
        for i, e in enumerate(items):
            setattr(self, e, i)

    def __call__(self):
        return self.__items

    # operator [] overloading
    def __getitem__(self, key):
        return self.__items[key]

    # define print format
    def __repr__(self):
        return '[' + ', '.join(self.__items) + ']'


class Cord:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __call__(self):
        return (self.x, self.y)

    def __repr__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

    def flatten_index(self, size: int):
        return self.x * size + self.y  # index of flatted array


def string_to_Cord(s):
    # TODO: Error Exception: using try & catch

    ts = literal_eval(s)
    if isinstance(ts, set):
        # ts is set of tuple(s)
        cs = []
        for t in ts:
            cs.append(Cord(t[0], t[1]))
        return set(cs)
    else:
        # ts is a single tuple
        ts = literal_eval(s)
        return Cord(ts[0], ts[1])
