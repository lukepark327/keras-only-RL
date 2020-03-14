
class Enum:
    def __init__(self, items: list):
        self._items = items
        self.size = len(items)
        for i, e in enumerate(items):
            setattr(self, e, i)

    def __call__(self):
        return self._items

    # operator [] overloading
    def __getitem__(self, key):
        return self._items[key]

    # define print format
    def __repr__(self):
        return '[' + ', '.join(self._items) + ']'
