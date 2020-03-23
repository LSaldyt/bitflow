from ..utils.module import Module

import pickle

import matplotlib.pyplot as plt

from uuid import uuid4

DPI  = 60
SIZE = 226

class AirfoilPlotter(Module):
    '''
    Plot a mined airfoil
    '''
    def __init__(self, name='AirfoilPlotter'):
        Module.__init__(self, in_label='Airfoil', out_label='AirfoilPlot:Image', connect_labels=('image', 'image'), name=name)

    def process(self, node, driver=None):
        coord_file  = node.data['coord_file']
        with open(coord_file, 'rb') as infile:
            coordinates = pickle.load(infile)
        fx, fy, sx, sy, camber = coordinates
        plt.plot(fx, fy, color='black')
        plt.plot(sx, sy, color='black')
        plt.axis('off')
        filename = 'data/images/' + node.data['name'] + str(uuid4()) + '.png'
        figsize  = (SIZE/DPI, SIZE/DPI)
        plt.savefig(filename, figsize=figsize, dpi=DPI)
        yield self.default_transaction(data=dict(filename=filename, parent=str(node.uuid)))


