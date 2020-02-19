from ..utils.OnlineLearner import OnlineLearner
from ..libraries.efficient_net.species_model import SpeciesModel

import torch
import os

class SpeciesClassifier(OnlineLearner):
    '''
    Classify species using a convolutional neural network
    '''
    def __init__(self, filename='data/models/species_classifier.nn'):
        OnlineLearner.__init__(self, in_label='Image', name='SpeciesClassifier', filename=filename)

    def init_model(self):
        self.model = SpeciesModel()

    def save(self):
        torch.save(self.model.state_dict(), self.filename)

    def load(self):
        try:
            self.model.load_state_dict(torch.load(self.filename)) # Takes roughly .15s
        except RuntimeError:
            os.remove(self.filename + '.bak') # Removes old backup!
            os.rename(self.filename, self.filename + '.bak')

    def learn(self, batch):
        print(batch, flush=True)
