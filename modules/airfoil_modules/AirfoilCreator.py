from .AirfoilRegressor import AirfoilRegressor, AirfoilModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pprint import pprint

class AirfoilCreator(AirfoilRegressor):
    '''
    Create an airfoil geometry based on desired performance coefficients
    '''
    def __init__(self, filename='data/models/airfoil_creator.nn'):
        AirfoilRegressor.__init__(self, filename=filename)

    def init_model(self):
        self.model = AirfoilModel(4 + 3 + 3, 1000) # Reverse of AirfoilRegressor's default

    def transform(self, node):
        coordinates, coefficient_tuples, coefficient_keys, alphas, limits, regime_vec = self.read_node(node)
        coordinates = sum(map(list, coordinates), [])
        for alpha, coefficients, (top, bot) in zip(alphas, coefficient_tuples, limits):
            inputs  = torch.Tensor(list(coefficients) + regime_vec + [top, bot, alpha])
            outputs = torch.Tensor(coordinates)
            break
            yield inputs, outputs



