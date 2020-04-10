from pprint import pprint
from subprocess import call
from time import time, sleep

import requests, zipfile, os
from neo4j import GraphDatabase, basic_auth

from time import sleep
import shutil, os

from petal.pipeline.module_utils.module import Module

NEO4J_IMPORT_DIR = '../../.Neo4jDesktop/neo4jDatabases/database-c60209c1-62b4-4cb3-90be-59a9b62b4141/installation-4.0.2/import/'


class DirectoryParser(Module):
    '''
    Populate neo4j with NASA directory info
    '''
    def __init__(self, import_dir=NEO4J_IMPORT_DIR, in_label=None, out_label='DirectorParserFinishedSignal', connect_label=None, name='Directory Parser'):
        Module.__init__(self, in_label, out_label, connect_label, name)
        self.import_dir = import_dir

    def read(self, filename, headers_index=0):
        with open(filename, 'r') as infile:
            residuals = None
            for i, line in enumerate(infile):
                cells = [cell.strip() for cell in line.split(',')]
                if i == headers_index:
                    yield cells
                elif i > headers_index:
                    if residuals is None:
                        residuals = cells
                    else:
                        residuals = [r if cell.strip() == '' else cell for cell, r in zip(cells, residuals)]
                    yield residuals

    def long_form(self, filename, headers_index=0):
        reader  = self.read(filename, headers_index=headers_index)
        headers = next(reader)
        data    = {h : [] for h in headers}
        for row in reader:
            for label, cell in zip(headers, row):
                data[label].append(cell)
        return data

    def process(self, driver=None):
        self.driver = self.get_driver(driver=driver)

        directory    = 'data/directory_data/'
        people       = self.long_form(directory + 'people.csv', headers_index=1)
        projects     = self.long_form(directory + 'projects.csv')
        data_science = self.long_form(directory + 'data_science.csv')

        for science_type, subtype, application in zip(data_science['Type'], data_science['subtype'], data_science['application type']):
            print(science_type, subtype, application, flush=True)
            yield self.custom_transaction(data=dict(name=science_type), in_label=None, out_label='DataScienceType', uuid=science_type + '_DataScienceType')
            # yield self.custom_transaction(data=dict(name=subtype), in_label='DataScienceType', out_label='DataScienceSubType', uuid=subtype + '_DataScienceSubType', from_uuid=science_type + '_DataScienceType', connect_labels=('subtype', 'subtype'))
            # yield self.custom_transaction(data=dict(name=application), in_label='DataScienceSubType', out_label='Application', uuid=application + '_Application', from_uuid=subtype + '_DataScienceSubType', connect_labels=('application', 'application'))
        print('Done', flush=True)

        # with self.driver.neo_client.session() as session:
        #     with open('data/cache/catalog.csv', 'r') as infile:
        #         headers = infile.readline().split(',')
        #     try:
        #         session.run('CREATE INDEX ON :Taxon(taxonRank)')
        #     except neobolt.exceptions.ClientError:
        #         pass
        #     try:
        #         session.run('CREATE INDEX ON :Taxon(name)')
        #     except neobolt.exceptions.ClientError:
        #         pass
        #     print('Adding catalog.csv')
        #     session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///catalog.csv" AS line CREATE (x:Taxon {' + ','.join(h + ': line.' + h for h in headers) + '})')
        #     print('Adding relations.csv')
        #     session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///relations.csv" AS line MATCH (x:Taxon {name: line.from}),(y:Taxon {name: line.to}) CREATE (x)-[:supertaxon]->(y)')

        # yield self.default_transaction(data=dict(done=True), uuid='__optimized_catalog_finished_signal__')
        # print('Optimized Catalog finished')
