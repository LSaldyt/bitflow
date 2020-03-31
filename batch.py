import pickle
from uuid import uuid4

class Batch:
    def __init__(self, label, uuid=None):
        self.items    = []
        self.length   = 0
        self.label    = label
        if uuid is None:
            self.uuid = str(uuid4())
        else:
            self.uuid = uuid
        self.filename = 'data/batches/' + str(uuid)

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.append(item)
        self.length += 1

    def save(self):
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self.items, outfile)

    def load(self):
        with open(self.filename, 'rb') as infile:
            self.items = pickle.load(infile)

    def clear(self):
        del self.items[:]
