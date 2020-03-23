from ..utils.OnlineTorchLearner import OnlineTorchLearner

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

from pprint import pprint

import json, os, os.path, pickle

from time import sleep
from PIL import Image

class EdgeRegressorModel(nn.Module):
    def __init__(self, depth=1, kernel_size=3, start_channels=4, mid_channels=10, pool_size=2, activation=nn.ReLU):
        nn.Module.__init__(self)
        self.mid_channels = mid_channels
        self.conv_layers = []
        for i in range(depth):
            if i == 0:
                in_channels = start_channels
            else:
                in_channels = mid_channels

            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size),
                nn.MaxPool2d(pool_size, pool_size),
                activation()
                ))
        self.final = nn.Sequential(nn.Linear(mid_channels * 26 * 26, 100), nn.Linear(100, 1000))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, 26 * 26 * self.mid_channels)
        x = self.final(x)
        x = x.double().squeeze(dim=0)
        return x

class AirfoilEdgeRegressor(OnlineTorchLearner):
    def __init__(self, filename='data/models/airfoil_edge_regressor.nn', name='AirfoilEdgeRegressor'):
        self.driver = None
        optimizer_kwargs = dict(lr=0.0001, momentum=0.9)
        OnlineTorchLearner.__init__(self, nn.MSELoss, optim.SGD, optimizer_kwargs, in_label='AirfoilPlot', name=name, filename=filename)

    def load_image(self, node):
        filename = node.data['filename']
        tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img = tfms(Image.open(filename))
        img = img.unsqueeze(0)
        return img

    def load_labels(self, node):
        parent = self.driver.get(node.data['parent'])
        with open(parent['coord_file'], 'rb') as infile:
            coordinates = pickle.load(infile)
        coordinates = sum(map(list, coordinates), [])
        return torch.tensor(coordinates, dtype=torch.double)

    def init_model(self):
        self.model = EdgeRegressorModel(depth=3)

    def transform(self, node):
        labels = self.load_labels(node)
        image  = self.load_image(node)
        yield image, labels

    def process(self, node, driver=None):
        if self.driver is None:
            self.driver = driver[0](driver[1])
        if os.path.isfile(self.filename):
            self.load()
        self.learn(node)
        self.save()
        return []
