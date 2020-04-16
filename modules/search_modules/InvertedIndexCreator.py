from petal.pipeline.module_utils.module import Module
from ..libraries.natural_language.hitlist import HitList

class InvertedIndexCreator(Module):
    '''
    Create an inverted index from hitlists
    '''
    def __init__(self, in_label='HitList', out_label=None, connect_labels=None, name='InvertedIndexCreator'):
        Module.__init__(self, in_label, out_label, connect_labels, name, page_batches=True)

    def process(self, previous):
        data = previous.data
