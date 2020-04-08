from petal.pipeline.module_utils.module import Module

class MockEOLSpeciesModule(Module):
    def __init__(self, in_label=None, out_label='EOLPage', connect_labels=None, name='EOLSpecies'):
        Module.__init__(self, in_label, out_label, connect_labels, name, page_batches=False)

    def process(self, driver=None):
        taxon_properties = {'kingdom'     : 'Animalia', 
                            'phylum'      : 'Chordata', 
                            'class'       : 'Vertebrata', 
                            'order'       : 'Mammals', 
                            'superfamily' : '', 
                            'family'      : 'Rorquals', 
                            'genus'       : 'Megaptera', 
                            'subgenus'    : '', 
                            'species'     : 'Megaptera novaeangliae'}
        name = taxon_properties['species']
        yield self.custom_transaction(data=taxon_properties, in_label=None, out_label='Taxon', uuid=name, connect_labels=None)
        data = {'canonical' : name, 'page_id' : 46559443, 'rank' : 'genus'}
        yield self.custom_transaction(data=data, in_label='Taxon', out_label='EOLPage', uuid=str(data['page_id']), from_uuid=name, connect_labels=('eol_page', 'eol_page'))

