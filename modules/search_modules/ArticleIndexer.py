from petal.pipeline.module_utils.module import Module
from ..libraries.natural_language.cleaner import Cleaner

class HitList:
    def __init__(self):
        self.sections = dict()
        self.words    = set()

    def add(self, section, word):
        self.words.add(word)

        if section not in self.sections:
            self.sections[section] = dict()
        section_counter = self.sections[section]
        if word in section_counter:
            section_counter[word] += 1
        else:
            section_counter[word] = 1


class ArticleIndexer(Module):
    '''
    This module is intended to index articles within PeTaL
    '''
    def __init__(self, in_label='Article', out_label='HitList', connect_labels=('hitlist', 'hitlist'), name='ArticleIndexer'):
        Module.__init__(self, in_label, out_label, connect_labels, name, page_batches=True)
        self.SECTIONS = {'title', 'summary'}
        self.cleaner = Cleaner()

    def process(self, previous):
        self.log.log('Running Indexer')

        hitlist = HitList()

        data = previous.data
        print(data['title'], flush=True)
        for section in self.SECTIONS:
            text = data[section]
            for word in self.cleaner.clean(text):
                hitlist.add(section, word)

        yield self.default_transaction(data=hitlist.sections)
