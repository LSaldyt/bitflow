
import os
import json
from .utils import get_module_names, fetch

def create_dependencies(directory='modules'):
    '''
    Read pipeline modules in a directory and get their type signatures
    Dump these to a JSON file.

    **Potentially uses a lot of memory, and is separated for this reason**

    '''
    dependencies = dict()
    for name in get_module_names(directory=directory):
        module = fetch(name, directory=directory)
        in_type  = module.in_label
        out_type = module.out_label
        page_batches = module.page_batches
        dependencies[name] = (in_type, out_type, page_batches)
    with open('.dependencies.json', 'w') as outfile:
        json.dump(dependencies, outfile, indent=4)

if __name__ == '__main__':
    create_dependencies()
