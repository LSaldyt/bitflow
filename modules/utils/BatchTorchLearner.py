from .BatchLearner import BatchLearner

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

from time import sleep

import plotly.graph_objs as go

class BatchTorchLearner(BatchLearner):
    '''
    Base class for pytorch machine learning modules
    '''
    def __init__(self, criterion=None, optimizer=None, optimizer_kwargs=None, **kwargs):
        BatchLearner.__init__(self, **kwargs)
        self.criterion = criterion()
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)

    def save(self):
        torch.save(self.model.state_dict(), self.filename)

    def load(self):
        try:
            self.model.load_state_dict(torch.load(self.filename)) # Takes roughly .15s
        except RuntimeError:
            backup = self.filename + '.bak'
            if os.path.isfile(backup):
                os.remove(backup) # Removes old backup!
            os.rename(self.filename, backup)
        except PermissionError:
            sleep(1)
        except FileNotFoundError:
            self.log.log('Weight file {} not found, starting from scratch'.format(self.filename))

    def transform(self, node):
        '''
        Must yield a list of tuples of (inputs, labels) for training
        '''
        pass

    def step(self, inputs, labels):
        raise RuntimeError('Batch learner called step()')

    def learn(self, batch):
        self.log.log('Learning')
        input_list = []
        label_list = []
        for node in batch.items:
            for inputs, labels in self.transform(node):
                input_list.append(inputs)
                label_list.append(labels)
        self.optimizer.zero_grad()
        inputs = torch.cat(input_list)
        labels = torch.cat(label_list)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        print(loss.item(), flush=True)
        self.log.log('{} loss: '.format(self.name), loss.item(), flush=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1], y=[loss.item()]))
        fig.show()
        
