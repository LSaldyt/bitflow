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
