from ..utils.BatchTorchLearner import BatchTorchLearner

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

import json, os, os.path, pickle

from time import sleep
from PIL import Image
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

class EdgeRegressorModel(nn.Module):
    def __init__(self, depth=1, activation=nn.ReLU, out_size=120, mid_channels=8):
        nn.Module.__init__(self)
        # self.norm = nn.BatchNorm2d(4)
        self.conv_layers = []
        for i in range(depth):
            if i == 0:
                channels = 4
            else:
                channels = 8
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(channels, mid_channels, 3, padding=1),
                # nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                # nn.BatchNorm2d(mid_channels),
                activation()
                ))
            if i < 2:
                self.conv_layers.append(nn.MaxPool2d(2))
        self.prefinal = nn.Sequential(
            nn.Linear(56 * 56 * mid_channels, 1000),
            )
        self.final = nn.Sequential(nn.Linear(1000, out_size))

    def forward(self, x):
        # x = self.norm(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, 56 * 56 * 8)
        x = self.prefinal(x)
        x = self.final(x)
        x = x.double()
        return x

class AirfoilEdgeRegressor(BatchTorchLearner):
    def __init__(self, filename='data/models/airfoil_edge_regressor.nn', name='AirfoilEdgeRegressor'):
        BatchTorchLearner.__init__(self, filename=filename, epochs=2, train_fraction=0.8, test_fraction=0.15, validate_fraction=0.05, criterion=nn.MSELoss, optimizer=optim.Adadelta, optimizer_kwargs=dict(lr=1.0, rho=0.9, eps=1e-06, weight_decay=0), in_label='AugmentedAirfoilPlot', name=name)

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
        fx = [x for i, x in enumerate(fx) if i % 5 == 0]
        fy = [y for i, y in enumerate(fy) if i % 5 == 0]
        sy = [y for i, y in enumerate(sy) if i % 5 == 0]
        coordinates = sum(map(list, [fx, fy, sy]), [])
        labels = torch.tensor(coordinates, dtype=torch.double)
        return labels.unsqueeze(0)

    def init_model(self):
        self.model = EdgeRegressorModel(depth=3)

    # def learn() inherited, uses transform()
    def transform(self, node):
        labels = self.load_labels(node.data['parent'])
        image  = self.load_image(filename = node.data['filename'])
        yield image, labels

    def test(self, batch):
        self.log.log('Testing on ')
        for node in batch.items:
            image  = self.load_image(filename=node.data['filename'])
            coordinates = self.model(image).detach().numpy()[0]
            figsize  = (800/DPI, 200/DPI)
            plt.figure(figsize=figsize, dpi=DPI)
            fx = coordinates[:40]
            fy = coordinates[40:80]
            sy = coordinates[80:120]
            fx = smooth(fx, 2)
            fy = smooth(fy, 2)
            sy = smooth(sy, 2)
            plt.plot(fx, fy, color='red')
            plt.plot(fx, sy, color='blue')
            plt.plot([fx[0], fx[0]], [sy[0], fy[0]], color='black') # Connect front
            plt.plot([fx[-1], fx[-1]], [sy[-1], fy[-1]], color='black') # Connect back

            parent = node.data['parent']
            parent = self.driver.get(parent)
            with open(parent['coord_file'], 'rb') as infile:
                base_coords = pickle.load(infile)
            fx, fy, sx, sy, camber = base_coords
            plt.plot(fx, fy, color='black')
            plt.plot(sx, sy, color='black')
            plt.plot([sx[0], fx[0]], [sy[0], fy[0]], color='black') # Connect front
            plt.plot([sx[-1], fx[-1]], [sy[-1], fy[-1]], color='black') # Connect back
            plt.axis('off')
            plt.show()

    def val(self, batch):
        self.log.log('Validating on ', batch.uuid)
