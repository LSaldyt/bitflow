from multiprocessing import Process, Queue, Pool, Manager
from queue import Empty
from collections import defaultdict
from collections import Counter
from time import sleep, time
from uuid import uuid4

import json

from .utils.utils import fetch
from .batch import Batch
from .driver import Driver, driver_listener
from .module_utils.log import Log

def save_batch(schedule_queue, transaction_queue, batch):
    batch.save()
    batch.clear()
    transaction_queue.put(batch)
    schedule_queue.put(batch)

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
            save_batch(schedule_queue, transaction_queue, batch)
            batches.pop(label)
        duration = time() - start
        i += 1

def run_module(module, serialize_queue, batch, driver=None):
    if batch is None:
        module.log.log('Backbone run ', module.name)
        gen = module.process(driver=driver)
    else:
        module.log.log('Batched returning run with ', module.name)
        batch.load()
        gen = module.process_batch(batch, driver=driver)
    if gen is not None:
        for transaction in gen:
            serialize_queue.put(transaction)
    module.log.log('Finished queueing transactions from ', module.name)

def module_runner(module_name, serialize_queue, batch, driver=None):
    with fetch(module_name) as module:
        run_module(module, serialize_queue, batch, driver=driver)

def pager(name, label, serialize_queue, driver_creator, delay, page_size):
    log = Log(name=name, directory='paging')
    driver_constructor, settings_file = driver_creator
    driver = driver_constructor(settings_file)
    module = fetch(name)

    batch_counts = Counter()
    matcher = 'MATCH (n:Batch) WHERE n.label = \'{}\' '.format(label)

    while True:
        count = next(driver.run_query(matcher + 'WITH COUNT (n) AS count RETURN count').records())['count']
        if count > 0:
            for i in range(count // page_size):
                page_query = matcher + 'RETURN (n) SKIP {} LIMIT {}'.format(i * page_size, page_size)
                pages = driver.run_query(page_query).records()
                for page in pages:
                    label    = page['n']['label']
                    uuid     = page['n']['uuid']
                    rand     = page['n']['rand']
                    if batch_counts[uuid] < module.epochs:
                        batch_counts[uuid] += 1
                        log.log('Running page: ', str(uuid))
                        batch = Batch(label, uuid=uuid, rand=rand)
                        run_module(module, serialize_queue, batch, driver=driver)
                    else:
                        pass
        sleep(delay)

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
        self.driver_process.daemon = True
        self.sizes  = self.settings['batch_sizes']
        self.limits = self.settings['process_limits']
        self.indep_batch_process     = Process(target=batch_serializer, args=(self.indep_serialize_queue, self.transaction_queue, self.schedule_queue, self.sizes))
        self.indep_batch_process.daemon = True
        self.batch_process     = Process(target=batch_serializer, args=(self.serialize_queue, self.transaction_queue, self.schedule_queue, self.sizes))
        self.batch_process.daemon = True
        self.dependents        = defaultdict(list)
        self.workers           = []
        self.pagers            = []
        self.waiting           = []
        with open('.dependencies.json', 'r') as infile:
            self.dependencies = json.load(infile)
        self.driver_creator = (Driver, settings_file)
        self.log = Log('scheduler')

    def schedule(self, module_name):
        self.log.log('Scheduling ', module_name)
        in_label, out_label, page = self.dependencies[module_name]
        if page:
            self.log.log('Paging database for ', module_name)
            self.pagers.append(Process(target=pager, args=(module_name, in_label, self.serialize_queue, self.driver_creator, self.settings['pager_delay'], self.settings['page_size'])))
            self.pagers[-1].daemon = True
            self.add_dependents(in_label, module_name)
        elif in_label is None:
            self.log.log('Starting ', module_name)
            proc = Process(target=module_runner, args=(module_name, self.indep_serialize_queue, None, self.driver_creator))
            proc.daemon = True
            self.workers.append((module_name, proc))
        else:
            self.log.log('Added dependent: ', module_name)
            self.add_dependents(in_label, module_name)

    def add_dependents(self, in_label, module_name):
        for label in in_label.split(','):
            for sublabel in label.split(':'):
                self.dependents[sublabel].append(module_name)

    def start(self):
        self.driver_process.start()
        self.batch_process.start()
        self.indep_batch_process.start()
        for name, process in self.workers:
            process.start()
        for pager in self.pagers:
            pager.start()

    def stop(self):
        self.driver_process.terminate()
        self.batch_process.terminate()
        self.indep_batch_process.terminate()
        for name, process in self.workers:
            process.terminate()
        for pager in self.pagers:
            pager.terminate()

    def add_proc(self, dep_proc):
        dependent, process = dep_proc
        self.log.log('Starting dependent ', dependent, ' ', process)
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
                batch = self.schedule_queue.get(block=False)
                for sublabel in batch.label.split(':'):
                    for dependent in self.dependents[sublabel]:
                        proc = Process(target=module_runner, args=(dependent, self.serialize_queue, batch, self.driver_creator))
                        proc.daemon = True
                        dep_proc = (dependent, proc)
                        if len(self.workers) < self.max_workers and self.check_limit(dependent):
                            self.add_proc(dep_proc)
                        else:
                            self.waiting.append(dep_proc)
            else:
                break
        return False

    def status(self, duration):
        running = dict()
        for dep, _ in self.workers:
            if dep in running:
                running[dep] += 1
            else:
                running[dep] = 1
        waiting_counts = dict()
        for dep, _ in self.waiting:
            if dep in waiting_counts:
                waiting_counts[dep] += 1
            else:
                waiting_counts[dep] = 1

        running_str = ' '.join('{} ({})'.format(dep, count) for dep, count in sorted(running.items(), key=lambda t : t[0]))
        waiting_str = ' '.join('{} ({})'.format(dep, count) for dep, count in sorted(waiting_counts.items(), key=lambda t : t[0]))
        queue_str   = 'transactions : {}, scheduled : {}, waiting : {}'.format(self.transaction_queue.qsize(), self.schedule_queue.qsize(), len(self.waiting))
        self.log.log('STATUS {}s'.format(round(duration, 2)))
        self.log.log('  RUNNING {}'.format(running_str))
        self.log.log('  WAITING {}'.format(waiting_str))
