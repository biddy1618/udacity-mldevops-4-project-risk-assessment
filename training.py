'''
Module for training an ML model.

Author: Dauren Baitursyn
Date: 10.09.22
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression
import json
import pickle

from pathlib import Path

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path.joinpath(Path.cwd(), config['output_folder_path'])
model_path = Path.joinpath(Path.cwd(), config['output_model_path'])


def train_model():
    '''Function for training the model.'''

    df = pd.read_csv(Path.joinpath(dataset_csv_path, 'finaldata.csv'))

    # Categorical variable `corporation` is unique, thus we drop it
    # cat_fields = ['corporation']
    num_fields = [
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees']
    target = 'exited'

    y = df.loc[:, target].values.ravel()
    X = df.loc[:, num_fields].values

    lg = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='auto', n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False)
    lg.fit(X, y)

    with open(Path.joinpath(model_path, 'trainedmodel.pkl'), 'wb') as f:
        pickle.dump(lg, f)


if __name__ == '__main__':
    train_model()
