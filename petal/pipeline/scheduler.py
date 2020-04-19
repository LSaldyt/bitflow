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
    transaction_queue.put(batch)
    schedule_queue.put(batch)

def batch_serializer(serialize_queue, transaction_queue, schedule_queue, sizes):
    start = time()
    batches = dict()
    i = 0
    while True:
        try:
            transaction = serialize_queue.get(block=False)
            label = transaction.out_label
            if label not in batches:
                batches[label] = Batch(label)
            batch = batches[label]
            batch.add(transaction)
            max_length = sizes.get(label, sizes['__default__'])
            if len(batch) >= max_length:
                save_batch(schedule_queue, transaction_queue, batches.pop(label))
        except:
            for label, batch in batches.items():
                if len(batch) > 0:
                    save_batch(schedule_queue, transaction_queue, batch)
            batches = dict()
        duration = time() - start
        i += 1

def run_module(module, serialize_queue, batch):
    module.log.log('Initiated ', module.name, ' run_module() in scheduler')
    if batch is None:
        module.log.log('Backbone run ', module.name)
        gen = module.process()
    else:
        module.log.log('Batched returning run with ', module.name)
        gen = module.process_batch(batch)
    if gen is not None:
        for transaction in gen:
            module.log.log(transaction)
            serialize_queue.put(transaction)
    module.log.log('Finished queueing transactions from ', module.name)

def module_runner(module_name, serialize_queue, batch, settings_file, module_dir='modules'):
    with fetch(module_name, directory=module_dir) as module:
        module.add_driver(Driver(settings_file))
        run_module(module, serialize_queue, batch)

def pager(name, label, serialize_queue, settings_file, delay, page_size, module_dir='modules'):
    log = Log(name=name, directory='paging')
    driver = Driver(settings_file)
    module = fetch(name, directory=module_dir)
    module.add_driver(driver)

    batch_counts = Counter()
    matcher = 'MATCH (n:Batch) WHERE n.label = \'{}\' '.format(label)

    while True:
        query = matcher + 'WITH COUNT (n) AS count RETURN count'
        count = next(driver.run_query(query).records())['count']
        log.log('Paging using query: ', query)
        log.log(name, ' page count: ', count)
        if count > 0:
            log.log('Continuing')
            for i in range(count // page_size + 1):
                page_query = matcher + 'RETURN (n) SKIP {} LIMIT {}'.format(i * page_size, page_size)
                log.log('Page query: ', page_query)
                pages = driver.run_query(page_query).records()
                for page in pages:
                    label    = page['n']['label']
                    uuid     = page['n']['uuid']
                    rand     = page['n']['rand']
                    has_epochs = hasattr(module, 'epochs')
                    max_count = module.epochs if has_epochs else 1
                    if batch_counts[uuid] < max_count:
                        batch_counts[uuid] += 1
                        log.log('Running page: ', str(uuid))
                        batch = Batch(label, uuid=uuid, rand=rand)
                        batch.load()
                        run_module(module, serialize_queue, batch)
                    else:
                        pass
        sleep(delay)

class Scheduler:
    def __init__(self, settings_file, module_dir):
        self.module_dir = module_dir
        with open(settings_file, 'r') as infile:
            self.settings = json.load(infile)
        self.settings_file = settings_file
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
        self.dependents        = defaultdict(set)
        self.workers           = []
        self.pagers            = []
        self.waiting           = []
        with open('.dependencies.json', 'r') as infile:
            self.dependencies = json.load(infile)
        self.log = Log('scheduler')

    def schedule(self, module_name):
        self.log.log('Scheduling ', module_name)
        in_label, out_label, page = self.dependencies[module_name]
        if page:
            self.log.log('Paging database for ', module_name)
            self.pagers.append(Process(target=pager, args=(module_name, in_label, self.serialize_queue, self.settings_file, self.settings['pager_delay'], self.settings['page_size'], self.module_dir)))
            self.pagers[-1].daemon = True
            self.add_dependents(in_label, module_name)
        elif in_label is None:
            proc = Process(target=module_runner, args=(module_name, self.indep_serialize_queue, None, self.settings_file, self.module_dir))
            proc.daemon = True
            self.workers.append((module_name, proc))
        else:
            self.add_dependents(in_label, module_name)

    def add_dependents(self, in_label, module_name):
        if in_label is not None:
            for label in in_label.split(','):
                for sublabel in label.split(':'):
                    if module_name not in self.dependents[sublabel]:
                        self.log.log('Added dependent: ', module_name)
                        self.dependents[sublabel].add(module_name)

    def start(self):
        self.driver_process.start()
        self.batch_process.start()
        self.indep_batch_process.start()
        for name, process in self.workers:
            self.log.log('Starting ', name)
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
                        proc = Process(target=module_runner, args=(dependent, self.serialize_queue, batch, self.settings_file, self.module_dir))
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
        # self.log.log('STATUS {}s'.format(round(duration, 2)))
        # self.log.log('  RUNNING {}'.format(running_str))
        # self.log.log('  WAITING {}'.format(waiting_str))
