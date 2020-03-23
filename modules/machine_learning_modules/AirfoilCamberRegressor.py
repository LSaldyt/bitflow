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

import matplotlib.pyplot as plt

class AirfoilCamberRegressor(AirfoilRegressor):
    '''
    Regress the performance of airfoil geometries, augmented with camber line and relative thickness
    '''
    def __init__(self, filename='data/models/airfoil_camber_regressor.nn'):
        AirfoilRegressor.__init__(self, filename=filename, name='AirfoilCamberRegressor')

    def init_model(self):
        self.model = AirfoilModel(1000 + 3 + 3, 4)

    def calculate_camber_augmentation(self, coordinates, plot=False):
        fx, sx, fy, sy = coordinates
        camber = [(fyi + syi) / 2.0 for fyi, syi in zip(fy, sy)]
        if plot:
            plt.plot(fx, fy, color='blue')
            plt.plot(sx, sy, color='red')
            plt.plot(fx, camber, color='orange')
            plt.show()
        return camber

    def transform(self, node):
        coordinates, coefficient_tuples, alphas, limits, regime_vec = self.read_node(node)
        coordinates += (self.calculate_camber_augmentation(coordinates),)
        coordinates = sum(map(list, coordinates), [])
        for alpha, coefficients, (top, bot) in zip(alphas, coefficient_tuples, limits):
            coefficients = torch.Tensor(coefficients)
            inputs       = torch.Tensor(coordinates + regime_vec + [top, bot, alpha])
            yield inputs, coefficients
