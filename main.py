#!/usr/bin/env python
from pipeline import PipelineInterface
import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        settings_file = 'config/default.json'
    else:
        settings_file = args[0]
    print('LOADING PeTaL config ({})'.format(settings_file), flush=True)
    # interface = PipelineInterface(settings_file, module_dir='modules')
    interface = PipelineInterface(settings_file, module_dir='test_modules')
    interface.log.log('Loaded settings from ', settings_file)
    interface.start_server(clean=True)
