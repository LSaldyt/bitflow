from .AirfoilEdgeRegressor import AirfoilEdgeRegressor
from ..utils.module        import Module

class AirfoilTester(AirfoilEdgeRegressor):
    def __init__(self):
        AirfoilEdgeRegressor.__init__(self)
        Module.__init__(self, in_label=None, out_label='BioAirfoil')
        self.load() # Load trained weights

    def process(self):
        image = self.get_image('data/whale_flipper_cross_section.png')
        coordinates = self.model(image)
        print(coordinates)
