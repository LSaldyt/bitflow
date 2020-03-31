import pickle

class Batch:
    def __init__(self, label):
        self.items    = []
        self.length   = 0
        self.label    = label
        self.filename = None

    def __len__(self):
        return self.length

    def add(self, item):
        self.items.append(item)
        self.length += 1

    def save(self, filename):
        self.filename = filename
        with open(filename, 'wb') as outfile:
            pickle.dump(self.items, outfile)

    def load(self, filename):
        self.filename = filename
        with open(filename, 'rb') as infile:
            self.items = pickle.load(infile)

    def clear(self):
        del self.items[:]
        self.length = 0
        self.filename = None
