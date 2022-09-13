'''
Script for running the whole process.

Author: Dauren Baitursyn
Date: 13.09.22
'''
import json
from pathlib import Path
from ast import literal_eval

import training
import scoring
import deployment
import diagnostics
import reporting

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = Path.joinpath(Path.cwd(), config['input_folder_path'])
output_folder_path = Path.joinpath(Path.cwd(), config['output_folder_path'])

with open(Path.joinpath(output_folder_path, 'ingestedfiles.txt'), 'r') as f:
    ingested_files = literal_eval(f.read())

##################Check and read new data
#first, read ingestedfiles.txt


#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt



##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







