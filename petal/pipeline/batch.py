import pickle
from uuid import uuid4
from random import random

class Batch:
    def __init__(self, label, uuid=None, rand=None):
        self.items    = []
        self.label    = label
        if uuid is None:
            self.uuid = str(uuid4())
        else:
            self.uuid = uuid
        self.filename = 'data/batches/' + str(self.uuid)
        if rand is None:
            self.rand = random()
        else:
            self.rand = rand

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.append(item)

    # def save(self):
    #     with open(self.filename, 'wb') as outfile:
    #         pickle.dump(self.items, outfile)

    # def load(self):
    #     with open(self.filename, 'rb') as infile:
    #         self.items = pickle.load(infile)
    #     # try:
    #     # except FileNotFoundError as e:
    #     #     print('Batch was loaded as empty:')
    #     #     print(e, flush=True)

    # def clear(self):
    #     del self.items[:]
