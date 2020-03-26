from .AirfoilEdgeRegressor import AirfoilEdgeRegressor
from ..utils.module        import Module

import matplotlib.pyplot as plt

class AirfoilTester(AirfoilEdgeRegressor):
    def __init__(self):
        AirfoilEdgeRegressor.__init__(self)
        Module.__init__(self, in_label=None, out_label='BioAirfoil')
        self.load() # Load trained weights

    def process(self, driver=None):
        image = self.load_image('data/whale_flipper_cross_section.png')
        # image = self.load_image('data/images/a18-il - A18 (original)5a5ac876-5915-443b-8986-8fa3017b36c7_augmented_0_9.png')
        coordinates = self.model(image).detach().numpy()
        
        # coordinates = sum(map(list, [fx, fy, sy]), [])
        fx = coordinates[:20]
        fy = coordinates[20:40]
        sy = coordinates[40:60]

        plt.plot(fx, fy, color='red')
        plt.plot(fx, sy, color='blue')
        plt.show()
        yield self.default_transaction(dict())
