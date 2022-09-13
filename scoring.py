'''
Module for scoring ML model.

Author: Dauren Baitursyn
Date: 10.09.22
'''
import json
import pickle

import pandas as pd
from pathlib import Path
from sklearn import metrics

with open('config.json', 'r') as f:
    config = json.load(f)

prod_path = Path.joinpath(Path.cwd(), config['prod_deployment_path'])
model_path = Path.joinpath(Path.cwd(), config['output_model_path'])
test_path = Path.joinpath(Path.cwd(), config['test_data_path'])
output_path = Path.joinpath(Path.cwd(), config['output_folder_path'])


def score_model(production=False):
    '''
    Function for model scoring.

    Returns:
        float: F1 score of the deployed model.
    '''
    num_fields = [
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees']
    target = 'exited'

    if production:
        df = pd.read_csv(Path.joinpath(output_path, 'finaldata.csv'))
    else:
        df = pd.read_csv(Path.joinpath(test_path, 'testdata.csv'))

    y_test = df.loc[:, target].values.ravel()
    X_test = df.loc[:, num_fields].values
    if production:
        with open(Path.joinpath(prod_path, 'trainedmodel.pkl'), 'rb') as f:
            lb = pickle.load(f)
    else:
        with open(Path.joinpath(model_path, 'trainedmodel.pkl'), 'rb') as f:
            lb = pickle.load(f)

    y_pred = lb.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred)
    with open(Path.joinpath(model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(f1_score))

    return f1_score


if __name__ == '__main__':
    score_model()
