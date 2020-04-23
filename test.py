import pickle

from pprint import pprint

with open('data/index', 'rb') as infile:
    data = pickle.load(infile)
    pprint(data)
    pprint(data['megaptera'])


