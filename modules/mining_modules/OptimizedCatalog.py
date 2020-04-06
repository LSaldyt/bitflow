from pprint import pprint
from subprocess import call
from time import time, sleep

import requests, zipfile, os
from neo4j import GraphDatabase, basic_auth

from time import sleep
import shutil, os

from ..utils.module import Module

from .NormalCatalog import create_dir, to_json

'''
This is the backbone mining module for population neo4j with the initial species list
'''

def to_long_json():
    for json in to_json():
        found = False
        relations = []
        for taxon in ['species', 'subgenus', 'genus', 'family', 'superfamily', 'order', 'class', 'phylum', 'kingdom']:
            if found:
                if json[taxon].strip() != '':
                    relations.append((json['name'], json[taxon]))
                    break
            rank = json['taxonRank']
            if rank == 'infraspecies':
                if taxon == 'species': # Necessary
                    found = True
            elif taxon == rank:
                found = True
        yield json, relations

def to_csv():
    i = 0
    first = True
    with open('data/cache/catalog.csv', 'w', encoding='utf-8') as catalog:
        with open('data/cache/species.csv', 'w', encoding='utf-8') as species_csv:
            with open('data/cache/relations.csv', 'w', encoding='utf-8') as relations:
                for entry, rels in to_long_json():
                    if i % 1000 == 0:
                        print(i, flush=True)
                    i += 1
                    if first:
                        catalog.write(','.join(entry.keys()) + '\n')
                        species_csv.write(','.join(entry.keys()) + '\n')
                        relations.write('from,to\n')
                        first = False
                    if entry['taxonRank'] == 'species':
                        species_csv.write(','.join(entry.values()) + '\n')
                    else:
                        catalog.write(','.join(entry.values()) + '\n')
                    if len(rels) > 0:
                        relations.write('\n'.join(','.join(r) for r in rels) + '\n')


class OptimizedCatalog(Module):
    '''
    Populate Taxa into the database in an optimized manor
    '''
    def __init__(self, import_dir='../../.Neo4jDesktop/neo4jDatabases/database-956b6711-76c3-46c6-80aa-4f335d68b2f8/installation-3.5.14/import/', in_label=None, out_label='CatalogFinishedSignal', connect_label=None, name='OptimizedCatalog'):
        Module.__init__(self, in_label, out_label, connect_label, name)
        self.import_dir = import_dir

    def process(self, driver=None):
        if not os.path.isfile('data/cache/catalog.csv') or not os.path.isfile('data/cache/relations.csv') or not os.path.isfile('data/cache/species.csv'):
            to_csv()
        for filename in os.listdir('.'):
            if filename.endswith('.csv'):
                shutil.copy(filename, self.import_dir + filename)
        driver = self.get_driver(driver=driver)

        with driver.neo_client.session() as session:
            headers = 'id,identifier,datasetID,datasetName,acceptedNameUsageID,parentNameUsageID,taxonomicStatus,taxonRank,verbatimTaxonRank,scientificName,kingdom,phylum,class,order,superfamily,family,genericName,genus,subgenus,specificEpithet,infraspecificEpithet,scientificNameAuthorship,source,namePublishedIn,nameAccordingTo,modified,description,taxonConceptID,scientificNameID,references,name'.split(',')
            session.run('CREATE INDEX ON :Taxon(name)')
            print('Adding catalog.csv')
            session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///catalog.csv" AS line CREATE (x:Taxon {' + ','.join(h + ': line.' + h for h in headers) + '})')
            print('Adding species.csv')
            session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///species.csv" AS line CREATE (x:Species:Taxon {' + ','.join(h + ': line.' + h for h in headers) + '})')
            print('Adding relations.csv')
            session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///relations.csv" AS line MATCH (x:Taxon {name: line.from}),(y:Taxon {name: line.to}) CREATE (x)-[:supertaxon]->(y)')

        sleep(100)
        yield self.default_transaction(data=dict(done=True))
        print('Optimized Catalog finished')
