from ..utils.OnlineLearner import OnlineLearner
from ..libraries.airfoil_regression.airfoil_model import AirfoilModel

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

import os

class AirfoilRegressor(OnlineLearner):
    '''
    Classify species using a convolutional neural network
    '''
    def __init__(self, filename='data/models/airfoil_regressor.nn'):
        OnlineLearner.__init__(self, in_label='Airfoil', name='AirfoilRegressor', filename=filename)
        self.init_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9) # TODO: Change me later!
        self.labels = dict()
        self.index  = 0

    def init_model(self):
        self.model = AirfoilModel(420, 69)

    def save(self):
        torch.save(self.model.state_dict(), self.filename)

    def load(self):
        try:
            self.model.load_state_dict(torch.load(self.filename)) # Takes roughly .15s
        except RuntimeError:
            backup = self.filename + '.bak'
            if os.path.isfile(backup):
                os.remove(backup) # Removes old backup!
            os.rename(self.filename, backup)

    def learn(self, node):
        print('Running airfoil regressor on:')
        print(node, flush=True)

        coord_file  = node.data['coord_file']
        detail_file = node.data['detail_file']

        with open(coord_file, 'rb') as infile:
            coordinates = pickle.load(infile)
        with open(detail_file, 'rb') as infile:
            details = pickle.load(infile)

        print(list(details.keys()), flush=True)
        print(coordinates)
        print(len(coordinates[0]))

