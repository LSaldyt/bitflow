from ..utils.OnlineLearner import OnlineLearner
from ..libraries.airfoil_regression.airfoil_model import AirfoilModel

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

from pprint import pprint

import os
import math

class AirfoilRegressor(OnlineLearner):
    '''
    Classify species using a convolutional neural network
    '''
    def __init__(self, filename='data/models/airfoil_regressor.nn'):
        OnlineLearner.__init__(self, in_label='Airfoil', name='AirfoilRegressor', filename=filename)
        self.init_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9) # TODO: Change me later!

    def init_model(self):
        self.model = AirfoilModel(800 + 3 + 3, 4)

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

    def read_node(self, node):
        coord_file  = node.data['coord_file']
        detail_file = node.data['detail_file']

        with open(coord_file, 'rb') as infile:
            coordinates = pickle.load(infile)
        with open(detail_file, 'rb') as infile:
            details = pickle.load(infile)

        signed_log = lambda x : 0 if x == 0 else math.copysign(math.log(abs(x)), x)

        mach  = signed_log(node.data['mach'])
        Re    = signed_log(node.data['Re'])
        Ncrit = signed_log(node.data['Ncrit'])
        regime_vec = [mach, Re, Ncrit]

        coefficient_tuples = list(zip(*(details[k] for k in sorted(details.keys()) if k.startswith('C'))))
        alphas = details['alpha']
        limits = list(zip(details['Top_Xtr'], details['Bot_Xtr']))

        return coordinates, coefficient_tuples, alphas, limits, regime_vec

    def step(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def learn(self, node):
        coordinates, coefficient_tuples, alphas, limits, regime_vec = self.read_node(node)
        coordinates = sum(map(list, coordinates), [])
        for alpha, coefficients, (top, bot) in zip(alphas, coefficient_tuples, limits):
            coefficients = torch.Tensor(coefficients)
            inputs       = torch.Tensor(coordinates + regime_vec + [top, bot, alpha])
            loss = self.step(inputs, coefficients)
            print('Regressor loss: ', loss, flush=True)


