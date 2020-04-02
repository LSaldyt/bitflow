from .AirfoilCreator import AirfoilCreator

import pickle
import matplotlib.pyplot as plt

class AirfoilCreatorTester(AirfoilCreator):
    def __init__(self):
        AirfoilCreator.__init__(self)
        self.load() # Load trained weights

    def process(self, driver=None):
        print('Showing off AirfoilCreator', flush=True)
        # self.model = AirfoilModel(4 + 3 + 3, 1000) # Reverse of AirfoilRegressor's default
