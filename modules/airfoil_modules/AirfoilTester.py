from .AirfoilEdgeRegressor import AirfoilEdgeRegressor
from ..utils.module        import Module

import matplotlib.pyplot as plt

class AirfoilTester(AirfoilEdgeRegressor):
    def __init__(self):
        AirfoilEdgeRegressor.__init__(self)
        Module.__init__(self, in_label=None, out_label='BioAirfoil')
        self.load() # Load trained weights

    def process(self, driver=None):
        # image = self.load_image('data/whale_flipper_cross_section.png')
        image = self.load_image('data/whale_flipper_cross_section.png')
        # image = self.load_image('data/images/a63a108c-il - NASA_AMES 63A108 MOD C AIRFOILbc67474a-07c2-4f33-8ac6-edca1bc47d31.png')
        # image = self.load_image('data/images/ag18-il - AG18017793c3-8288-4efd-bfd7-aaa888559cbb.png')
        image = self.load_image('data/images/a18-il - A18 (original)e9ba9a40-206b-498e-a488-64643232ab4e.png')
        coordinates = self.model(image).detach().numpy()
        
        # coordinates = sum(map(list, [fx, fy, sy]), [])
        fx = coordinates[:200]
        fy = coordinates[200:400]
        sy = coordinates[400:600]

        plt.plot(fx, fy, color='black')
        plt.plot(fx, sy, color='black')
        plt.show()
        yield self.default_transaction(dict())
