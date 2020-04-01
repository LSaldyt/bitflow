from importlib import import_module
import json, os


def get_module_subdirs():
    for name in os.listdir('modules'):
        if 'modules' in name:
            yield name

def fetch(module_name):
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        for subdir in get_module_subdirs():
            try:
                # print('modules.{}.{}'.format(subdir, module_name))
                module = import_module('modules.{}.{}'.format(subdir, module_name))
                break
            except ModuleNotFoundError:
                pass
        else:
            raise RuntimeError('Could not find module: ' + module_name)
    return getattr(module, module_name)()
            
def get_module_names():
    for subdir in get_module_subdirs():
        modules = os.listdir('modules/' + subdir)
        for filename in modules:
            if filename.endswith('.py') and filename != '__init__.py':
                name = os.path.basename(filename).split('.')[0]
                yield filename.replace('.py', '')
