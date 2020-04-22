import pickle, os
from random import random

def clean(item):
    if item is None:
        return None
    item = str(item)
    item = item.replace(' ', '_')
    item = item.replace('-', '_')
    item = item.replace('\\', '_')
    item = item.replace('/', '_')
    item = item.replace('\'', '')
    item = item.replace('(', '')
    item = item.replace(')', '')
    return item

class Batch:
    def __init__(self, label, uuid=None, rand=None):
        self.do_save     = False
        self.items    = []
        self.label    = label
        if uuid is None:
            raise ValueError('Batch was supplied with UUID None')
        self.uuid = clean(uuid)
        self.filename = 'data/batches/' + str(self.uuid)
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
            self.do_save = True

    def save(self):
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self.items, outfile)

    def load(self):
        if os.path.isfile(self.filename):
            with open(self.filename, 'rb') as infile:
                self.items = pickle.load(infile)
        else:
            raise OSError('No batch file ' + self.filename)
