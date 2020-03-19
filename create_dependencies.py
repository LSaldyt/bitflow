
import os
from utils.utils import get_module_names, fetch

def create_dependencies():
    for name in get_module_names():
        print(name)
        module = fetch(name)
        print(module)

if __name__ == '__main__':
    create_dependencies()
