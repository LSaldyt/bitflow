from ..utils.module import Module

class MockEOLSpeciesModule(Module):
    def __init__(self, in_label=None, out_label='EOLPage', connect_labels=None, name='EOLSpecies', count=1900000):
        Module.__init__(self, in_label, out_label, connect_labels, name, count)

    def process(self):
        # x.canonical, x.page_id, x.rank
        data = {'x.canonical' : 'Megaptera', 'x.page_id' : 46559443, 'x.rank' : 'genus'}
        data = {k.replace('x.', '') : v for k, v in data.items()}
        name = 'Megaptera'
        transaction = self.custom_transaction(data=data)
        yield transaction
        # yield self.custom_transaction(data=data, in_label='Taxon', out_label='EOLPage', uuid=name + '_eol_page', from_uuid=name, connect_labels=('eol_page', 'eol_page'))

