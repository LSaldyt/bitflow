from importlib import import_module
import json, os

'''
Random code that has no better home. Most of this is meta relative to the bitflow
'''

EXCLUDE = {'libraries', '__init__.py', '__pycache__'}

def get_module_subdirs(directory='modules'):
    for name in os.listdir(directory):
        if name not in EXCLUDE:
            yield name

def fetch(module_name, directory='modules', settings_file=None):
    for subdir in get_module_subdirs(directory=directory):
        for filename in os.listdir(directory + '/' + subdir):
            if module_name in filename:
                name = directory + '.{}.{}'.format(subdir, module_name)
                module = import_module(name)
                return getattr(module, module_name)()
    raise ModuleNotFoundError('Could not find module: ' + module_name)
            
def get_module_names(directory='modules'):
    for root, folders, files in os.walk(directory):
        if not ('libraries' in root or '__pycache__' in root):
        for filename in files:
            if filename != '__init__.py':
                yield f.replace('.py', '')

def clean_uuid(item):
    '''
    Used on UUIDs and links
    '''
    if item is None:
        return None
    item = str(item)
    # item = item.replace(' ', '_')
    item = item.replace('-', '_')
    item = item.replace('\\', '_')
    item = item.replace('/', '_')
    item = item.replace('\'', '')
    item = item.replace('(', '')
    item = item.replace(')', '')
    return item

def add_json_node(tx, label='Generic', properties=None):
    if properties is None:
        properties = dict()
    prop_set = '{' + ','.join('{key}:${key}'.format(key=k) for k in properties) + '}'
    query = 'MERGE (n:{label} '.format(label=label) + prop_set + ') RETURN n'
    return tx.run(query, **properties)
