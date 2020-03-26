from ..utils.OnlineTorchLearner import OnlineTorchLearner

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

from PIL import Image

from pprint import pprint

from efficientnet_pytorch import EfficientNet as EfficientNetBase

import json, os, os.path

from time import sleep


class HierarchicalModel(nn.Module):
    def __init__(self, i=0, outputs=1, width=1):
        nn.Module.__init__(self)
        if i < 0 or i > 7:
            raise ValueError('Parameter i to Efficient Net Model must be between 0 and 7 inclusive, but was: {}'.format(i))
        # Top-1 Accuracy ranges from 76.3% to 84.4%, in intervals of roughly 1-2% between indexes
        self.feature_extractor = EfficientNetBase.from_pretrained('efficientnet-b{}'.format(i)) # Can go up to b7, with b0 having the least parameters, and b7 having the most (but more accuracy)
        self.fc1 = nn.Linear(1280 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 500)

        self.outputs = [nn.Linear(500, width) for _ in range(outputs)]

    def forward(self, x):
        x = self.feature_extractor.extract_features(x)
        x = x.view(1280 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return tuple(final_layer(x) for final_layer in self.outputs)

TAXA = {'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'genus', 'subgenus', 'species'}

class TaxonClassifier(OnlineTorchLearner):
    '''
    Classify taxa using a convolutional neural network
    '''
    def __init__(self, filename='data/models/taxon_classifier.nn'):
        OnlineTorchLearner.__init__(self, nn.CrossEntropyLoss, optim.SGD, dict(lr=0.001, momentum=0.9), in_label='Image', name='TaxonClassifier', filename=filename)
        self.init_model()
        self.driver = None

        self.label_map    = {taxa : {'' : 0} for taxa in TAXA} # Map empty str to 0
        self.label_counts = {taxa : 1 for taxa in TAXA}

    def load_image(self, node):
        filename = node.data['filename']
        tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]) # Explanation of these magic numbers??
        img = tfms(Image.open(filename))
        img = img.unsqueeze(0)
        return img

    def init_model(self):
        self.model = HierarchicalModel(outputs=len(TAXA), width=100)
        # self.model = HierarchicalModel(outputs=1, width=2)

    def load_labels(self, node):
        parent = self.driver.get(node.data['parent'])
        name = parent['canonical']
        taxon  = self.driver.get(name)
        labels = []
        for taxa in TAXA:
            taxa_name = taxon[taxa]
            sub_label_map = self.label_map[taxa]
            if taxa_name not in sub_label_map:
                sub_label_map[taxa_name] = self.label_counts[taxa]
                self.label_counts[taxa] += 1
            labels.append(torch.tensor([sub_label_map[taxa_name]], dtype=torch.long))
        return labels

    def transform(self, node):
        print('TaxonClassifier:', node, flush=True)
        labels = self.load_labels(node)
        image  = self.load_image(node)
        yield image, labels

    def step(self, inputs, labels):
        losses = []
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = None
        for output, label in zip(outputs, labels):
            output = output.unsqueeze(dim=0)
            if loss is None:
                loss = self.criterion(output, label)
            else:
                loss += self.criterion(output, label)
            losses.append(loss.item())
        loss.backward()
        self.optimizer.step()
        return losses

    def learn(self, node):
        for inputs, labels in self.transform(node):
            loss = self.step(inputs, labels)
            # pprint(self.label_map)
            print('{} loss: '.format(self.name), loss, flush=True)

    def process(self, node, driver=None):
        if self.driver is None:
            self.driver = driver[0](driver[1])
        if os.path.isfile(self.filename):
            self.load()
        self.learn(node)
        self.save()
        return []
