'''
Script for running the whole process.

Author: Dauren Baitursyn
Date: 13.09.22
'''
import sys
import json
import logging

from pathlib import Path
from ast import literal_eval

import reporting
from ingestion import ingest_data
from training import train_model
from scoring import score_model
from deployment import deploy_model
from apicalls import api_returns

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

with open('config.json', 'r') as f:
    config = json.load(f)

prod_path = Path.joinpath(Path.cwd(), config['prod_deployment_path'])
input_path = Path.joinpath(Path.cwd(), config['input_folder_path'])


# Check and read new data
logger.info('Checking for new data...')
with open(Path.joinpath(prod_path, 'ingestedfiles.txt'), 'r') as f:
    ingested_files = literal_eval(f.read())

source_files = list(f.name for f in input_path.iterdir())
new_files = False
if not set(ingested_files).issuperset(source_files):
    logger.info(
        f'Found new data: {set(source_files).difference(ingested_files)}')
    new_files = True

# Deciding whether to proceed
if not new_files:
    logger.info('No new data. Aborting.')
    exit(0)

ingest_data()

# Checking for model drift
logger.info('Checking for score change...')
with open(Path.joinpath(prod_path, 'latestscore.txt'), 'r') as f:
    f1_score_old = literal_eval(f.read())
f1_score_new = score_model(production=True)
# exit()

# Deciding whether to proceed, part 2
if f1_score_new > f1_score_old:
    logger.info('Model performs good on new data. Aborting.')
    # exit(0)

logger.info('Model performs worse on new data. Retraining.')
train_model()

# Re-deployment
logger.info('Deploying new model.')
deploy_model()

# Diagnostics and reporting
logger.info('Saving new confusion matrix.')
reporting.plot_confusion_matrix(production=True, name='confusionmatrix2.png')

logger.info('Testing API.')
api_returns(name='apireturns2.txt')
