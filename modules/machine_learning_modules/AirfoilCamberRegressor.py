from .AirfoilRegressor import AirfoilRegressor
from ..libraries.airfoil_regression.airfoil_model import AirfoilModel

from ..mining_modules.Airfoils import interpolate_airfoil

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

class AirfoilCamberRegressor(AirfoilRegressor):
    '''
    Regress the performance of airfoil geometries, augmented with camber line and relative thickness
    '''
    def __init__(self, filename='data/models/airfoil_camber_regressor.nn'):
        AirfoilRegressor.__init__(self, filename=filename, name='AirfoilCamberRegressor')

    def init_model(self):
        self.model = AirfoilModel(800 + 3 + 3, 4)

    def calculate_camber_augmentation(self):
        pass

    def transform(self, node):
        print('HERE', flush=True)
        coordinates, coefficient_tuples, alphas, limits, regime_vec = self.read_node(node)
        coordinates = sum(map(list, coordinates), [])
        for alpha, coefficients, (top, bot) in zip(alphas, coefficient_tuples, limits):
            coefficients = torch.Tensor(coefficients)
            inputs       = torch.Tensor(coordinates + regime_vec + [top, bot, alpha])
            yield inputs, coefficients
