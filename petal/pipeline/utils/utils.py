from importlib import import_module
import json, os

def get_module_subdirs(directory='modules'):
    for name in os.listdir(directory):
        if 'modules' in name:
            yield name

def fetch(module_name, directory='modules', driver_constructor=None, settings_file=None):
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        for subdir in get_module_subdirs(directory=directory):
            name = directory + '.{}.{}'.format(subdir, module_name)
            try:
                module = import_module(name)
                break
            except ModuleNotFoundError as e:
                pass
        else:
            raise RuntimeError('Could not find module: ' + name)
    return getattr(module, module_name)(driver_constructor, settings_file)
            
def get_module_names(directory='modules'):
    for subdir in get_module_subdirs(directory=directory):
        modules = os.listdir(directory + '/' + subdir)
        for filename in modules:
            if filename.endswith('.py') and filename != '__init__.py':
                name = os.path.basename(filename).split('.')[0]
                yield filename.replace('.py', '')
