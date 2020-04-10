from time import time, sleep
import json
import os
import sys
import shutil

from neo4j import GraphDatabase, basic_auth

from .utils.utils import get_module_names, fetch

from .scheduler import Scheduler
from .module_utils.log import Log
from .create_dependencies import create_dependencies


class PipelineInterface:
    '''
    This class defines an interface to a data mining server. It allows modules and settings to the scheduler to be updated dynamically without stopping processing.
    '''
    def __init__(self, filename, module_dir='modules'):
        self.module_dir = module_dir
        print('LOADING PeTaL config ({})'.format(filename), flush=True)
        create_dependencies(directory=module_dir)
        self.log = Log('pipeline_server')
        self.scheduler = Scheduler(filename, module_dir)
        self.times = dict()
        self.filename = filename
        self.sleep_time = 1
        self.reload_time = 30
        self.status_time = 1
        self.whitelist = []
        self.blacklist = []
        self.settings = self.load_settings()
        self.neo_client = GraphDatabase.driver(self.settings["neo4j_server"], auth=basic_auth(self.settings["username"], self.settings["password"]), encrypted=self.settings["encrypted"])

    def reload_modules(self):
        for name in get_module_names(directory=self.module_dir):
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
        return settings

    def start_server(self, clean=True):
        print('CLEANING Old Data', flush=True)
        if clean:
            self.clean()
        print('STARTING PeTaL Data Pipeline Server', flush=True)
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
                if duration > self.status_time:
                    self.scheduler.status(duration)
                if duration > self.reload_time:
                    start = time()
                    self.settings = self.load_settings()
                    self.reload_modules()
                    self.log.log('Actively reloading settings')
        except KeyboardInterrupt as interrupt:
            print('INTERRUPTING PeTaL Data Pipeline Server', flush=True)
        finally:
            print('STOPPING PeTaL Data Pipeline Server', flush=True)
            self.scheduler.stop()

    def clean(self):
        for directory in ['logs', 'profiles', 'batches', 'images']:
            fulldir = 'data/' + directory
            try:
                shutil.rmtree(fulldir)
            except FileNotFoundError:
                pass
            os.mkdir(fulldir)
            with open(fulldir + '/.placeholder', 'w') as outfile:
                outfile.write('')
        # with self.neo_client.session() as session:
        #     session.run('match (x)<-[r]->(y) delete r, x, y')
        #     session.run('match (n) delete n')
