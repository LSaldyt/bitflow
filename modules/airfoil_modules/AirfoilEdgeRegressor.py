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
    def __init__(self, depth=1, activation=nn.ReLU, out_size=60, mid_channels=8):
        nn.Module.__init__(self)
        self.norm = nn.BatchNorm2d(4)
        self.conv_layers = []
        for i in range(depth):
            if i == 0:
                channels = 4
            else:
                channels = 8
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(channels, mid_channels, 3, padding=1),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                nn.MaxPool2d(3, padding=1), 
                nn.BatchNorm2d(mid_channels),
                activation()
                ))
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3),
            activation()
            ))
        self.prefinal = nn.Sequential(
            nn.Linear(222 * 222 * mid_channels, 1000),
            )
        self.final = nn.Sequential(nn.Linear(1000, out_size))

    def forward(self, x):
        x = self.norm(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(222 * 222 * 8)
        x = self.prefinal(x)
        x = self.final(x)
        x = x.double().squeeze(dim=0)
        return x

class AirfoilEdgeRegressor(OnlineTorchLearner):
    def __init__(self, filename='data/models/airfoil_edge_regressor.nn', name='AirfoilEdgeRegressor'):
        self.driver = None
        OnlineTorchLearner.__init__(self, nn.MSELoss, optim.Adadelta, dict(lr=1.0, rho=0.9, eps=1e-06, weight_decay=0), in_label='AugmentedAirfoilPlot', name=name, filename=filename)

    def load_image(self, filename):
        tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        image = Image.open(filename)
        image.putalpha(255)
        img = tfms(image)
        img = img.unsqueeze(0)
        return img

    def load_labels(self, parent):
        parent = self.driver.get(parent)
        with open(parent['coord_file'], 'rb') as infile:
            coordinates = pickle.load(infile)
        fx, fy, sx, sy, camber = coordinates
        fx = [x for i, x in enumerate(fx) if i % 10 == 0]
        fy = [y for i, y in enumerate(fy) if i % 10 == 0]
        sy = [y for i, y in enumerate(sy) if i % 10 == 0]
        coordinates = sum(map(list, [fx, fy, sy]), [])
        return torch.tensor(coordinates, dtype=torch.double)

    def init_model(self):
        self.model = EdgeRegressorModel(depth=6)

    def transform(self, node):
        labels = self.load_labels(node.data['parent'])
        image  = self.load_image(filename = node.data['filename'])
        yield image, labels

    def process(self, node, driver=None):
        if self.driver is None:
            self.driver = driver[0](driver[1])
        if os.path.isfile(self.filename):
            self.load()
        self.learn(node)
        self.save()
        return []
