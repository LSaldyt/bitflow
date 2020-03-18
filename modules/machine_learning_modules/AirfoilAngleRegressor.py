from .AirfoilRegressor import AirfoilRegressor
from ..libraries.airfoil_regression.airfoil_model import AirfoilModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pprint import pprint

class AirfoilAngleRegressor(AirfoilRegressor):
    '''
    Guess the angle of attack based on geometry and metrics
    '''
    def __init__(self, filename='data/models/airfoil_angle_regressor.nn'):
        AirfoilRegressor.__init__(self, filename=filename)

    def init_model(self):
        self.model = AirfoilModel(4 + 3 + 2 + 800, 1) # Reverse of AirfoilRegressor's default

    def learn(self, node):
        coordinates, coefficient_tuples, alphas, limits, regime_vec = self.read_node(node)
        coordinates = sum(map(list, coordinates), [])
        for alpha, coefficients, (top, bot) in zip(alphas, coefficient_tuples, limits):
            inputs  = torch.Tensor(list(coefficients) + regime_vec + [top, bot] + coordinates)
            outputs = torch.Tensor([alpha])
            loss = self.step(inputs, outputs)
            print('Angle regressor: ', loss, flush=True)


