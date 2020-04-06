from .AirfoilCreator import AirfoilCreator
from petal.pipeline.module_utils.module import Module

import pickle, torch
import plotly.graph_objects as go

class AirfoilCreatorTester(AirfoilCreator):
    def __init__(self):
        AirfoilCreator.__init__(self)
        Module.__init__(self, in_label=None, out_label='Airfoil')
        self.load() # Load trained weights

    def process(self, driver=None):
        print('Showing off AirfoilCreator', flush=True)
        params = torch.tensor([7.2400e-02, 6.8460e-02, 1.4239e+00, 2.2900e-02, 0.0000e+00, 1.3816e+01,
        1.6094e+00, 1.1100e-02, 1.0000e+00, 1.9250e+01])
        coordinates = self.model(params).detach().numpy()

        fx     = coordinates[:20]
        fy     = coordinates[20:40]
        sx     = coordinates[40:60]
        sy     = coordinates[60:80]
        camber = coordinates[80:]

        print('Creating plotly figure', flush=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fx, y=fy, mode='lines', name='top'))
        fig.add_trace(go.Scatter(x=sx, y=sy, mode='lines', name='bottom'))
        fig.add_trace(go.Scatter(x=sx, y=camber, mode='lines', name='camber'))
        print('Added traces', flush=True)
        fig.update_layout(title='Creator Regression Test')
        fig.write_html('data/test.html', auto_open=True)
        print('Done w/ creator tester', flush=True)
