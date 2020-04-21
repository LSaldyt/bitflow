import pickle, os
from uuid import uuid4
from random import random

class Batch:
    def __init__(self, label, uuid=None, rand=None):
        self.save     = False
        self.items    = []
        self.label    = label
        if uuid is None:
            self.uuid = str(uuid4())
        else:
            self.uuid = uuid
        self.filename = 'data/batches/' + str(self.uuid.split('_')[0])
        if rand is None:
            self.rand = random()
        else:
            self.rand = rand

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.append(item)
        # If any contained transactions should be saved, then save them.
        if item.save: 
            self.save = True

    def save(self):
        print('Saving file to', self.filename)
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self.items, outfile)

    def load(self):
        print('Loading file from ', self.filename)
        if os.path.isfile(self.filename):
            with open(self.filename, 'rb') as infile:
                self.items = pickle.load(infile)
        else:
            raise OSError('No batch file ' + self.filename)
