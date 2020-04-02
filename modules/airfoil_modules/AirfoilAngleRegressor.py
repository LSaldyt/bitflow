from .AirfoilRegressor import AirfoilRegressor, AirfoilModel

import torch

class AirfoilAngleRegressor(AirfoilRegressor):
    '''
    Guess the angle of attack based on geometry and metrics
    '''
    def __init__(self, filename='data/models/airfoil_angle_regressor.nn'):
        AirfoilRegressor.__init__(self, filename=filename)

    def init_model(self):
        self.model = AirfoilModel(4 + 3 + 2 + 800, 1) # Reverse of AirfoilRegressor's default

    def transform(self, node):
        coordinates, coefficient_tuples, alphas, limits, regime_vec = self.read_node(node)
        coordinates = sum(map(list, coordinates), [])
        for alpha, coefficients, (top, bot) in zip(alphas, coefficient_tuples, limits):
            inputs  = torch.Tensor(list(coefficients) + regime_vec + [top, bot] + coordinates)
            outputs = torch.Tensor([alpha])
            yield inputs, outputs

