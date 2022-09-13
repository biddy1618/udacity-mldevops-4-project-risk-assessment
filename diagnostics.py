'''
Module for diagnostics metrics and stats.

Author: Dauren Baitursyn
Date: 11.09.22
'''

import os
import sys
import json
import timeit
import logging
import subprocess

import pandas as pd
from pickle import load
from pathlib import Path

with open('config.json', 'r') as f:
    config = json.load(f)

prod_path = Path.joinpath(Path.cwd(), config['prod_deployment_path'])
input_path = Path.joinpath(
    Path.cwd(), config['output_folder_path'])

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def model_predictions(data):
    '''
    Function to get model predictions.

    Args:
        data (pd.DataFrame): Data to perform predictions on trained model.

    Returns:
        list: List of predicitons.
    '''
    num_fields = [
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees']

    X = data.loc[:, num_fields].values

    with open(Path.joinpath(prod_path, 'trainedmodel.pkl'), 'rb') as f:
        model = load(f)

    pred = model.predict(X)

    return pred.tolist()


def dataframe_summary(data):
    '''
    Function to get summary statistics on numerical columns.

    Args:
        data (pd.DataFrame): Data to perform summary on.

    Returns:
        list[list]: List of statistics in order means, medians,
            and standard deviations.
    '''
    num_fields = [
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees']

    data_num = data.loc[:, num_fields]
    means = data_num.mean(axis=0).tolist()
    medians = data_num.median(axis=0).tolist()
    std_devs = data_num.std(axis=0).tolist()

    return [means, medians, std_devs]


def missing_data(data):
    '''
    Function to check for missing data.

    Args:
        data (pd.DataFrame): Data to perform summary on.

    Returns:
        list: Percentage of missing data per column.
    '''
    target = 'exited'
    data_ind = data.loc[:, data.columns != target]
    missing = (data_ind.isna().sum(axis=0) / len(data_ind)).tolist()

    return missing


def execution_time():
    '''
    Function to calculate timing of training and ingestion.

    Returns:
        list: List with two values for time executed for each task.
    '''
    start_timing = timeit.default_timer()
    os.system('python training.py')
    time_training = timeit.default_timer() - start_timing

    start_timing = timeit.default_timer()
    os.system('python ingestion.py')
    time_ingestion = timeit.default_timer() - start_timing

    return [time_training, time_ingestion]


def outdated_packages_list():
    '''
    Function to check dependencies.

    Returns:
        str: Packages with current and latest versions in the string format.
    '''
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    return str(outdated, 'utf-8')


if __name__ == '__main__':

    data = pd.read_csv(Path.joinpath(input_path, 'finaldata.csv'))

    pred = model_predictions(data)
    logger.info(f'Predictions by the model:\n{pred}\n')

    summary = dataframe_summary(data)
    logger.info(f'Summary statistics:\n{summary}\n')

    missing = missing_data(data)
    logger.info(f'Missing data:\n{missing}\n')

    timing = execution_time()
    logger.info(f'Time execution:\n{timing}\n')

    out_packages = outdated_packages_list()
    logger.info(f'Package versions:\n{out_packages}\n')
