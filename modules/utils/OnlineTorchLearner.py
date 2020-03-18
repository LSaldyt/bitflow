from .OnlineLearner import OnlineLearner

class OnlineTorchLearner(OnlineLearner):
    '''
    Base class for pytorch machine learning modules
    '''
    def __init__(self, criterion, optimizer, optimizer_kwargs, filename, in_label=None, out_label=None, name=None):
        # Take Airfoils as input, and produce no outputs.
        OnlineLearner.__init__(self, in_label=in_label, out_label=out_label, name=name, filename=filename)
        # Criteria needs to be MSE or anything compatible with regression
        self.criterion = criterion
        self.optimizer = optimizer(**optimizer_kwargs)

    def step(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def transform(self, node):
        '''
        Must yield a list of tuples of (inputs, labels) for training
        '''
        pass

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

    def learn(self, node):
        for inputs, labels in self.transform(node):
            loss = self.step(inputs, labels)
            print('{} loss: '.format(name), loss, flush=True)
