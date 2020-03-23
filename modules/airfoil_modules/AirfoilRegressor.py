from ..utils.OnlineTorchLearner import OnlineTorchLearner
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

class AirfoilRegressor(OnlineTorchLearner):
    '''
    Regress the performance of airfoil geometries
    Implements the `Module` interface, which requires a type signature and process() function.
    The `OnlineTorchLearner` class is defined in `utils`, and specifies common operations of online machine learning models
    To inherit from this class, AirfoilRegressor must specify `init_model`, and `transform`.
    Also, a filename can be specified to the parent constructor to specify where the model is saved.
    '''
    def __init__(self, name='AirfoilRegressor', filename='data/models/airfoil_regressor.nn'):
        # Take Airfoils as input, and produce no outputs.
        optimizer_kwargs = dict(lr=0.0001, momentum=0.9)
        OnlineTorchLearner.__init__(self, nn.MSELoss, optim.SGD, optimizer_kwargs, in_label='Airfoil', name=name, filename=filename)

    def init_model(self):
        self.model = AirfoilModel(1000 + 3 + 3, 4)

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

    def transform(self, node):
        coordinates, coefficient_tuples, alphas, limits, regime_vec = self.read_node(node)
        coordinates = sum(map(list, coordinates), [])
        for alpha, coefficients, (top, bot) in zip(alphas, coefficient_tuples, limits):
            coefficients = torch.Tensor(coefficients)
            inputs       = torch.Tensor(coordinates + regime_vec + [top, bot, alpha])
            yield inputs, coefficients
