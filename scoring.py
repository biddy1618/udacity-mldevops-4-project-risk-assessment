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

model_path = Path.joinpath(Path.cwd(), config['output_model_path'])
test_data_path = Path.joinpath(Path.cwd(), config['test_data_path'])


def score_model():
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

    test_df = pd.DataFrame()
    for f in test_data_path.iterdir():
        tmp_df = pd.read_csv(f)
        test_df = pd.concat([test_df, tmp_df])

    y_test = test_df.loc[:, target].values.ravel()
    X_test = test_df.loc[:, num_fields].values

    with open(Path.joinpath(model_path, 'trainedmodel.pkl'), 'rb') as f:
        lb = pickle.load(f)

    y_pred = lb.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred)
    with open(Path.joinpath(model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(f1_score))

    return f1_score


if __name__ == '__main__':
    score_model()
