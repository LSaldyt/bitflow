from time import time, sleep
from pprint import pprint
import json
import os
import sys
import shutil

from utils.utils import get_module_names, fetch

from scheduler import Scheduler
from log import Log

class PipelineInterface:
    '''
    This class defines an interface to a data mining server. It allows modules and settings to the scheduler to be updated dynamically without stopping processing.
    '''
    def __init__(self, filename):
        self.log = Log('pipeline_server')
        self.scheduler = Scheduler(settings_file)
        self.times = dict()
        self.filename = filename
        self.sleep_time = 1
        self.reload_time = 30
        self.whitelist = []
        self.blacklist = []
        self.load_settings()

    def reload_modules(self):
        for name in get_module_names():
            if len(self.whitelist) > 0:
                if name in self.whitelist:
                    self.scheduler.schedule(name)
            elif name not in self.blacklist:
                self.scheduler.schedule(name)

    def load_settings(self):
        with open(self.filename, 'r') as infile:
            settings = json.load(infile)
        self.log.log(settings)
        for k, v in settings.items():
            if k.startswith('scheduler:'):
                k = k.replace('scheduler:', '')
                setattr(self.scheduler, k, v)
            elif k.startswith('pipeline:'):
                k = k.replace('pipeline:', '')
                setattr(self, k, v)

    def start_server(self):
        self.log.log('Starting pipeline server')
        start = time()
        self.reload_modules() 
        self.log.log('Starting scheduler')
        self.scheduler.start()
        done = False
        try:
            while not done:
                done = self.scheduler.check()
                sleep(self.sleep_time)
                duration = time() - start
                if duration > self.reload_time:
                    start = time()
                    self.load_settings()
                    self.reload_modules()
                    self.log.log('Actively reloading settings')
        finally:
            print('STOPPING server!', flush=True)
            self.scheduler.stop()
            # shutil.rmtree('data/batches')
            # os.mkdir('data/batches')

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        settings_file = 'configurations/default.json'
    else:
        settings_file = args[0]
    interface = PipelineInterface(settings_file)
    interface.log.log('Loaded settings from ', settings_file)
    interface.start_server()
