import glob
import os
import time
import random
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

from ..utils.module import Module

airfoil_tools = "http://airfoiltools.com"
airfoil_search_url="http://airfoiltools.com/search/airfoils"

# Airfoil helper functions:

def scrape_airfoil_list():
    raw_html = get(airfoil_search_url).content
    html = BeautifulSoup(raw_html, 'html.parser')
    airfoilURLList = html.findAll("table", {"class": "listtable"})
    tableRows = airfoilURLList[0].findAll("tr")
    airfoil_urls = []
    airfoil_names = []
    for row in tableRows: # Search through all tables 
        airfoil_link = row.find(lambda tag: tag.name=="a" and tag.has_attr('href'))
        if (airfoil_link):
            airfoil_urls.append(airfoil_tools + airfoil_link['href'])
            airfoil_names.append(airfoil_link.text.replace("\\", "_").replace("/","_"))
    return airfoil_urls,airfoil_names

def scrape_airfoil_coords(airfoil_page,airfoilname):    
    lednicerDAT=airfoil_page.replace("details","lednicerdatfile")
    raw_html=get(lednicerDAT,True).content
    soup=BeautifulSoup(raw_html,'lxml')    
    with open('./scrape/{}.txt'.format(airfoilname), mode='wt', encoding='utf-8') as file:
        file.write(soup.text)

def scrape_details(details_page,airfoil_name,Re,Ncrit):
    raw_html=get(details_page).content
    html = BeautifulSoup(raw_html, 'html.parser')
    details_table = html.findAll("table", {"class": "details"})
    table_links = details_table[0].findAll("a")
    polar = table_links[2]['href']
    raw_html2 = get(airfoil_tools + polar,True).content
    with open('./scrape/{}.txt'.format(airfoil_name+"_polar_"+str(Re)+"_"+str(Ncrit)), mode='w') as file:
        file.write(raw_html2.decode('utf-8'))

def scrape_airfoil_polars(airfoil_page,airfoil_name):    
    raw_html=get(airfoil_page).content
    html = BeautifulSoup(raw_html, 'html.parser')
    polar_list = html.findAll("table", {"class": "polar"})
    tableRows = polar_list[0].findAll("tr")
    for row in tableRows: # Search through all rows
        columns = row.findAll("td")
        if (columns):
            if (len(columns)>4):
                Re = float(columns[2].text.replace(',',''))
                Ncrit = float(columns[3].text.replace(',',''))
                dataLink = columns[7].find(lambda tag: tag.name=="a" and tag.has_attr('href'))
                dataLink = dataLink['href']
                details_page = airfoil_tools + dataLink
                scrape_details(details_page,airfoil_name,Re,Ncrit)

class Airfoils(Module):
    '''
    '''
    def __init__(self, in_label=None, out_label='Airfoil', connect_labels=None, name='Airfoils'):
        Module.__init__(self, in_label, out_label, connect_labels, name)

    def process(self):
        airfoil_urls, airfoil_names = scrape_airfoil_list()
        l = len(airfoil_names)
        for i in range(0,len(airfoil_urls)):
            # print('{}/{}'.format(i, l), flush=True)
            # # Check if airfoil is already scraped 
            # if not os.path.isfile('scrape/' + airfoil_names[i] + ".txt"):
            #     scrape_airfoil_coords(airfoil_urls[i],airfoil_names[i])
            #     scrape_airfoil_polars(airfoil_urls[i],airfoil_names[i])
            yield self.default_transaction({'name' : airfoil_names[i]})

        
