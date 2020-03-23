from ..utils.module import Module

import pickle

import matplotlib.pyplot as plt

class AirfoilPlotter(Module):
    '''
    Plot a mined airfoil
    '''
    def __init__(self, name='AirfoilPlotter'):
        Module.__init__(self, in_label='Airfoil', out_label='AirfoilImage:Image', connect_labels=('image', 'image'), name=name)

    def transform(self, node):
        coord_file  = node.data['coord_file']
        with open(coord_file, 'rb') as infile:
            coordinates = pickle.load(infile)
        fx, sx, fy, sy, camber = coordinates
        print('plotting', flush=True)
        yield self.default_transaction(data=dict(filename='test'))
