from ..utils.OnlineTorchLearner import OnlineTorchLearner
from ..libraries.hierarchical_classifier.hierarchical_model import HierarchicalModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

from PIL import Image

import os

class TaxonClassifier(OnlineTorchLearner):
    '''
    Classify taxa using a convolutional neural network
    '''
    def __init__(self, filename='data/models/taxon_classifier.nn'):
        OnlineTorchLearner.__init__(self, nn.CrossEntropyLoss, optim.SGD, dict(lr=0.0001, momentum=0.9), in_label='Image', name='SpeciesClassifier', filename=filename)
        self.init_model()

    def load_image(self, filename):
        tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]) # Explanation of these magic numbers??
        img = tfms(Image.open(filename))
        img = img.unsqueeze(0)
        return img

    def init_model(self):
        self.model = HierarchicalModel()

    def transform(self, node):
        print('TaxonClassifier:', node, flush=True)
        return []
    # def learn(self, node):
    #     print('TaxonClassifier:', node, flush=True)
    #     return
    #     try:
    #         species  = node['parent']
    #         filename = node['filename']
    #         image    = self.load_image(filename)
    #         
    #         if species in self.labels:
    #             labels = self.labels[species]
    #         else:
    #             labels = self.index
    #             self.labels[species] = self.index
    #             self.index += 1

    #         labels = torch.tensor([labels], dtype=torch.long)
    #         inputs = image.squeeze(dim=1)

    #         self.optimizer.zero_grad()
    #         outputs = self.model(inputs)
    #         loss = self.criterion(outputs, labels)
    #         loss.backward()
    #         self.optimizer.step()
    #     except RuntimeError:
    #         pass
    #     except OSError:
    #         pass
