from ..utils.module import Module

from pprint import pprint

class MockMLModel(Module):
    def __init__(self, in_label='MLData', out_label=None, connect_labels=None, name='MockMLData', epochs=2):
        Module.__init__(self, in_label=in_label, out_label=out_label, connect_labels=connect_labels, name=name, page_batches=True)
        self.epochs = epochs

    def process(self, node, driver=None):
        pass

    def process_batch(self, batch, driver=None):
        print(batch.uuid, flush=True)

