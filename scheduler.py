from multiprocessing import Process, Queue, Pool, Manager
from queue import Empty
from collections import defaultdict
from time import sleep, time
from uuid import uuid4
from pprint import pprint

from driver import driver_listener
from batch import Batch

from utils.utils import fetch

import json

from driver import Driver

def save_batch(schedule_queue, transaction_queue, label, batch):
    filename = 'data/batches/{}'.format(uuid4())
    batch.save(filename)
    batch.clear()
    transaction_queue.put((label, filename))
    schedule_queue.put((label, filename))

def batch_serializer(serialize_queue, transaction_queue, schedule_queue, sizes):
    start = time()
    batches = dict()
    i = 0
    while True:
        transaction = serialize_queue.get()
        label = transaction.out_label
        if label not in batches:
            batches[label] = Batch(label)
        batch = batches[label]
        batch.add(transaction)
        max_length = sizes.get(label, sizes['__default__'])
        if len(batch) >= max_length:
            save_batch(schedule_queue, transaction_queue, label, batch)
        duration = time() - start
        i += 1


def module_runner(module_name, serialize_queue, batch_file, driver=None):
    module = fetch(module_name)

    if batch_file is None:
        gen = module.process(driver=driver)
    else:
        batch = Batch(module.in_label)
        batch.load(batch_file)
        gen = module.process_batch(batch, driver=driver)
    i = 0
    for transaction in gen:
        serialize_queue.put(transaction)
        i += 1

class Scheduler:
    def __init__(self, settings_file):
        with open(settings_file, 'r') as infile:
            self.settings = json.load(infile)
        self.max_workers = self.settings['scheduler:max_workers']
        self.transaction_queue = Queue()
        self.indep_serialize_queue = Queue()
        self.serialize_queue   = Queue()
        self.schedule_queue    = Queue()
        self.driver_process    = Process(target=driver_listener,  args=(self.transaction_queue, settings_file))
        self.sizes  = self.settings['batch_sizes']
        self.limits = self.settings['process_limits']
        self.indep_batch_process     = Process(target=batch_serializer, args=(self.indep_serialize_queue, self.transaction_queue, self.schedule_queue, self.sizes))
        self.batch_process     = Process(target=batch_serializer, args=(self.serialize_queue, self.transaction_queue, self.schedule_queue, self.sizes))
        self.dependents        = defaultdict(list)
        self.workers           = []
        self.waiting           = []
        with open('.dependencies.json', 'r') as infile:
            self.dependencies = json.load(infile)
        self.driver_creator = (Driver, settings_file)

    def schedule(self, module_name):
        print('Scheduling ', module_name, flush=True)
        in_label, out_label = self.dependencies[module_name]
        if in_label is None:
            print('Starting ', module_name, flush=True)
            self.workers.append((module_name, Process(target=module_runner, args=(module_name, self.indep_serialize_queue, None, self.driver_creator))))
        else:
            print('Added dependent: ', module_name, flush=True)
            self.dependents[in_label].append(module_name)

    def start(self):
        self.driver_process.start()
        self.batch_process.start()
        self.indep_batch_process.start()
        for name, process in self.workers:
            process.start()

    def stop(self):
        self.driver_process.terminate()
        self.batch_process.terminate()
        self.indep_batch_process.terminate()
        for name, process in self.workers:
            process.terminate()

    def add_proc(self, dep_proc):
        dependent, process = dep_proc
        print('Starting dependent', dependent, flush=True)
        process.start()
        self.workers.append((dependent, process))

    def check_limit(self, dependent):
        count = 0
        for name, worker in self.workers:
            if name == dependent:
                count += 1
        upper = self.limits.get(dependent, self.limits['__default__'])
        return count < upper

    def check(self):
        self.workers = [(name, worker) for name, worker in self.workers if worker.is_alive()]
        while len(self.waiting) > 0:
            dependent, proc = self.waiting[-1]
            if len(self.workers) < self.max_workers and self.check_limit(dependent):
                self.add_proc(self.waiting.pop())
            else:
                break
        while not self.schedule_queue.empty():
            if len(self.workers) < self.max_workers:
                label, batch_file = self.schedule_queue.get(block=False)
                if label is not None:
                    for sublabel in label.split(':'):
                        for dependent in self.dependents[sublabel]:
                            dep_proc = (dependent, Process(target=module_runner, args=(dependent, self.serialize_queue, batch_file, self.driver_creator)))
                            if len(self.workers) < self.max_workers and self.check_limit(dependent):
                                self.add_proc(dep_proc)
                            else:
                                self.waiting.append(dep_proc)
            else:
                break
        return False
