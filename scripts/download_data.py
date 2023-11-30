import os
import sys
import click
import json
import pandas as pd
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--write-to', type=str, help="Path to directory where raw data will be written to")

def main(url, write_to):
    """Downloads data from a URL and writes it to a specified directory"""
    df = download_data(url)
    df.to_pickle(write_to, compression='zip') 
    print("Downloading data from {} and writing it to {}".format(url, write_to))

def download_data(url):
    """Downloads data from a URL and converts it into a pandas dataframe"""
    print("Downloading data from url")
    resp = urlopen(url)
    myzip = ZipFile(BytesIO(resp.read()))

    data_list = []

    for line in myzip.open(myzip.namelist()[0]).readlines():
        data_list.append(json.loads(line))
    df = pd.DataFrame(data_list)
    return df

if __name__ == '__main__':
    main()