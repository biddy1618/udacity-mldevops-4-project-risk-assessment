'''
Module for ingestion.

Author: Dauren Baitursyn
Date: 10.09.22
'''

import os
import json

# from datetime import datetime
from pathlib import Path

import pandas as pd
# import numpy as np


with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = Path.joinpath(Path.cwd(), config['input_folder_path'])
output_folder_path = Path.joinpath(Path.cwd(), config['output_folder_path'])


def merge_multiple_dataframe():
    '''
    Read data from `input_folder_path` and save merged data to
    `output_folder_path` along with names of to-be ingested files.
    '''
    df = pd.DataFrame()
    ingested_files = []

    for f in os.listdir(input_folder_path):
        tmp_df = pd.read_csv(Path.joinpath(input_folder_path, f))
        df = df.append(tmp_df)
        ingested_files.append(f)

    df.drop_duplicates(inplace=True)

    df.to_csv(Path.joinpath(output_folder_path, 'finaldata.csv'), index=False)
    with open(Path.joinpath(output_folder_path, 'ingestedfiles.txt'), 'w') \
            as f:
        f.write(str(ingested_files))


if __name__ == '__main__':
    merge_multiple_dataframe()
