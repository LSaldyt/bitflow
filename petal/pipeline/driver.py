from neo4j import GraphDatabase, basic_auth
import json
import neobolt

from collections import defaultdict
from time import sleep, time
from queue import Empty

from .utils.neo import page, add_json_node
from .module_utils.log import Log
from .module_utils.profile import Profile
from .module_utils.transaction import Transaction
from .batch import Batch, clean

def retry(f):
    def inner(*args, **kwargs):
        waiting = True
        while waiting:
            try:
                return f(*args, **kwargs)
                waiting = False
            except neobolt.exceptions.ServiceUnavailable as e:
                print('Cannot reach neo4j server. Is it running? Sleeping 1s..', flush=True)
                sleep(1)
    return inner

class Driver():
    '''
    An API providing a lightweight connection to neo4j
    '''
    @retry
    def __init__(self, settings_file):
        with open(settings_file, 'r') as infile:
            settings = json.load(infile)
        self.neo_client = GraphDatabase.driver(settings["neo4j_server"], auth=basic_auth(settings["username"], settings["password"]), encrypted=settings["encrypted"])
        self.session = self.neo_client.session()
        self.hset = set()
        self.lset = set()

    @retry
    def run_query(self, query):
        return self.session.run(query)

    @retry
    def run(self, transaction):
        if transaction.query is not None:
            self.session.run(transaction.query)
        else:
            id1 = clean(transaction.from_uuid)
            id2 = clean(transaction.uuid)
            if transaction.data is not None:
                if id2 in self.hset:
                    return False
                self.hset.add(id2)
                self.add(transaction.data, transaction.out_label)
            if id1 is not None and transaction.connect_labels is not None:
                id1 = str(id1)
                key = str(id1) + str(id2)
                if key in self.lset:
                    return False
                self.lset.add(key)
                self.session.write_transaction(self._link, id1, id2, transaction.in_label, transaction.out_label, *transaction.connect_labels)
            return True

    def _link(self, tx, id1, id2, in_label, out_label, from_label, to_label):
        from_label = clean(from_label)
        to_label = clean(to_label)
        query = ('MATCH (n:{in_label}) WHERE n.uuid=\'{id1}\' MATCH (m:{out_label}) WHERE m.uuid=\'{id2}\''.format(in_label=in_label, out_label=out_label, id1=id1, id2=id2))
        if from_label is not None:
            query += (' MERGE (n)-[:{from_label}]->(m)'.format(from_label=from_label))
        if to_label is not None:
            query += (' MERGE (n)-[:{to_label}]->(m)'.format(to_label=to_label))
        tx.run(query)

    @retry
    def add(self, data, label):
        self.session.write_transaction(add_json_node, label, data)

    @retry
    def get(self, uuid):
        uuid = clean(uuid)
        records = list(self.session.run('MATCH (n) WHERE n.uuid = \'{uuid}\' RETURN n'.format(uuid=str(uuid))).records())
        if len(records) > 0:
            return records[0]['n']
        else:
            return None

    @retry
    def count(self, label):
        records = self.session.run('MATCH (x:{label}) WITH COUNT (x) AS count RETURN count'.format(label=label)).records()
        return list(records)[0]['count']

def driver_listener(transaction_queue, settings_file):
    profile = Profile('driver')
    log     = Log('driver')

    start = time()
    driver = Driver(settings_file)
    i = 0
    while True:
        batch = transaction_queue.get()
        for transaction in batch.items:
            log.log(transaction)
            try:
                added = driver.run(transaction)
            except TypeError as e:
                print(e)
                for k, v in transaction.data.items():
                    print(k)
                    print(type(v))
            if added:
                i += 1
        if batch.save and batch.label is not None:
            for sublabel in batch.label.split(':'):
                driver.run(Transaction(out_label='Batch', data={'label' : sublabel, 'filename' : batch.filename, 'rand' : batch.rand}, uuid=batch.uuid))
        duration = time() - start
        total = len(driver.hset) + len(driver.lset)
        log.log('Driver rate: {} of {} ({}|{})'.format(round(total / duration, 3), total, len(driver.hset), len(driver.lset)))
        log.log('Created batch for ', batch.label)
