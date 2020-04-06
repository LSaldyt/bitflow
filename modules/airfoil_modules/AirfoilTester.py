from .AirfoilEdgeRegressor import AirfoilEdgeRegressor
from petal.pipeline.module_utils.module import Module

import pickle
import plotly.graph_objects as go

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
        Module.__init__(self, in_label=None, out_label='FitAirfoil:Airfoil')
        self.load() # Load trained weights

    def plot(self, coordinates):
        fx = coordinates[:40]
        fy = coordinates[40:80]
        sy = coordinates[80:120]
        fx = smooth(fx, 2)
        fy = smooth(fy, 2)
        sy = smooth(sy, 2)
        print('Creating plotly figure', flush=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fx, y=fy, mode='lines', name='top'))
        fig.add_trace(go.Scatter(x=fx, y=sy, mode='lines', name='bottom'))
        print('Added traces', flush=True)
        fig.update_layout(title='Edge Regression Test')
        fig.write_html('data/tester.html', auto_open=True)
        print('Done', flush=True)

    def process(self, driver=None):
        # location = 'data/images/c141e-il - LOCKHEED C-141 BL761.11 AIRFOILfcafb389-f948-4050-a7cf-a1c23df4056d.png'
        location = 'data/whale_flipper_cross_section.png'
        # base = 'data/airfoil_data/wb140-il - WB-140_35_FB 14%_coords.pkl'
        # base = 'data/airfoil_data/2032c-il - 20-32C AIRFOIL_coords.pkl'
        # base = 'data/airfoil_data/c141e-il - LOCKHEED C-141 BL761.11 AIRFOIL_coords.pkl'
        
        image = self.load_image(location)
        coordinates = self.model(image).detach().numpy()[0]
        self.plot(coordinates)
        return []

        
