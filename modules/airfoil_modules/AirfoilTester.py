from .AirfoilEdgeRegressor import AirfoilEdgeRegressor
from ..utils.module        import Module

import pickle
import matplotlib.pyplot as plt

DPI = 400

def smooth(array, amount):
    new = []
    running = 0
    for i, x in enumerate(array):
        running += x
        if i % amount == 0:
            new.append(running/amount)
            running = 0
    return new

class AirfoilTester(AirfoilEdgeRegressor):
    def __init__(self):
        AirfoilEdgeRegressor.__init__(self)
        Module.__init__(self, in_label=None, out_label='BioAirfoil')
        self.load() # Load trained weights

    def process(self, driver=None):
        # location = 'data/images/wb140-il - WB-140_35_FB 14%bc5c2a11-84ea-418e-8385-c84917788dac_augmented_1_0.png'
        # location = 'data/images/wb140-il - WB-140_35_FB 14%bc5c2a11-84ea-418e-8385-c84917788dac.png'
        # location = 'data/images/2032c-il - 20-32C AIRFOIL2cbd1921-6f37-4ae3-a149-a88eeef1edae_augmented_0_7.png'
        # location = 'data/images/2032c-il - 20-32C AIRFOIL2cbd1921-6f37-4ae3-a149-a88eeef1edae.png'
        # location = 'data/images/ah80136-il - AH 80-136cfc274da-fff3-4cb6-bcc0-6e7bffe2459a_augmented_3_40.png'
        location = 'data/images/c141e-il - LOCKHEED C-141 BL761.11 AIRFOILfcafb389-f948-4050-a7cf-a1c23df4056d.png'
        base = 'data/airfoil_data/wb140-il - WB-140_35_FB 14%_coords.pkl'
        base = 'data/airfoil_data/2032c-il - 20-32C AIRFOIL_coords.pkl'
        # base = 'data/airfoil_data/ah80136-il - AH 80-136_coords.pkl'
        base = 'data/airfoil_data/c141e-il - LOCKHEED C-141 BL761.11 AIRFOIL_coords.pkl'
        
        # location = 'data/whale_flipper_cross_section.png'
        
        image = self.load_image(location)
        coordinates = self.model(image).detach().numpy()[0]

        
        figsize  = (800/DPI, 200/DPI)
        plt.figure(figsize=figsize, dpi=DPI)
        fx = coordinates[:40]
        fy = coordinates[40:80]
        sy = coordinates[80:120]
        # fx = smooth(fx, 5)
        # fy = smooth(fy, 5)
        # sy = smooth(sy, 5)
        plt.plot(fx, fy, color='red')
        plt.plot(fx, sy, color='blue')
        plt.plot([fx[0], fx[0]], [sy[0], fy[0]], color='black') # Connect front
        plt.plot([fx[-1], fx[-1]], [sy[-1], fy[-1]], color='black') # Connect back

        with open(base, 'rb') as infile:
            base_coords = pickle.load(infile)
        fx, fy, sx, sy, camber = base_coords
        plt.plot(fx, fy, color='black')
        plt.plot(sx, sy, color='black')
        plt.plot([sx[0], fx[0]], [sy[0], fy[0]], color='black') # Connect front
        plt.plot([sx[-1], fx[-1]], [sy[-1], fy[-1]], color='black') # Connect back
        plt.axis('off')
        plt.show()
        yield self.default_transaction(dict())
