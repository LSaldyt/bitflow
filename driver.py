from neo4j import GraphDatabase, basic_auth
from pprint import pprint
import json

from utils.neo import page, add_json_node
from uuid import uuid4
from collections import defaultdict
from time import sleep

class Driver():
    '''
    An API providing a lightweight connection to neo4j
    '''
    def __init__(self,):
        self.neo_client = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"), encrypted=False)
        # self.neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))

    def run(self, transaction):
        if transaction.query is not None:
            tx.run(transaction.query)
            return None
        else:
            id1 = transaction.uuid
            id2 = self.add(transaction.data, transaction.out_label)
            if id1 is not None and transaction.connect_labels is not None:
                id1 = str(id1)
                self.link(tx, id1, id2, transaction.in_label, transaction.out_label, *transaction.connect_labels)
            return id2

    def link(self, tx, id1, id2, in_label, out_label, from_label, to_label):
        query = ('MATCH (n:{in_label}) WHERE n.uuid=\'{id1}\' MATCH (m:{out_label}) WHERE m.uuid=\'{id2}\' MERGE (n)-[:{from_label}]->(m) MERGE (m)-[:{to_label}]->(n)'.format(in_label=in_label, out_label=out_label, id1=id1, id2=id2, from_label=from_label, to_label=to_label))
        tx.run(query)

    def add(self, data, label):
        with self.neo_client.session() as session:
            node = session.write_transaction(add_json_node, label, data)
            records = node.records()
            node = (next(records)['n'])
            id_n = node.id
            if 'uuid' not in node:
                unique_id = uuid4()
                session.run('MATCH (s) WHERE ID(s) = {} SET s.uuid = \'{}\' RETURN s'.format(id_n, str(unique_id)))
            else:
                unique_id = node['uuid']
            return unique_id

    def get(self, uuid):
        with self.neo_client.session() as session:
            records = list(session.run('MATCH (n) WHERE n.uuid = \'{uuid}\' RETURN n'.format(uuid=str(uuid))).records())
        if len(records) > 0:
            return records[0]
        else:
            raise ValueError('UUID {} invalid'.format(uuid))

    def count(self, label):
        with self.neo_client.session() as session:
            records = session.run('MATCH (x:{label}) WITH COUNT (x) AS count RETURN count'.format(label=label)).records()
        return list(records)[0]['count']

def driver_listener(transaction_queue, batch_queue):
    driver = Driver()
    while True:
        sleep(0.2)
        transaction = transaction_queue.get()
        print(transaction, flush=True)
        uuid = driver.run(transaction)
        node = driver.get(uuid)
        batch_queue.put(node)

