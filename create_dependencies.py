
import os
import json
from utils.utils import get_module_names, fetch

def create_dependencies():
    dependencies = dict()
    for name in get_module_names():
        module = fetch(name)
        in_type  = module.in_label
        out_type = module.out_label
        dependencies[name] = (in_type, out_type)
    with open('dependencies.json', 'w') as outfile:
        json.dump(dependencies, outfile, indent=4)

if __name__ == '__main__':
    create_dependencies()
