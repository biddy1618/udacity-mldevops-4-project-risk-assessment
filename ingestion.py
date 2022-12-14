'''
Module for ingestion.

Author: Dauren Baitursyn
Date: 10.09.22
'''
import json

from pathlib import Path

import pandas as pd


with open('config.json', 'r') as f:
    config = json.load(f)

input_path = Path.joinpath(Path.cwd(), config['input_folder_path'])
output_path = Path.joinpath(Path.cwd(), config['output_folder_path'])


def ingest_data():
    '''
    Read data from `input_folder_path` and save merged data to
    `output_folder_path` along with names of to-be ingested files.
    '''
    df = pd.DataFrame()
    ingested_files = []

    for f in input_path.iterdir():
        tmp_df = pd.read_csv(f)
        df = pd.concat([df, tmp_df])
        ingested_files.append(f.name)

    df.drop_duplicates(inplace=True)

    df.to_csv(Path.joinpath(output_path, 'finaldata.csv'), index=False)
    with open(Path.joinpath(output_path, 'ingestedfiles.txt'), 'w') \
            as f:
        f.write(str(ingested_files))

    return df


if __name__ == '__main__':
    ingest_data()
