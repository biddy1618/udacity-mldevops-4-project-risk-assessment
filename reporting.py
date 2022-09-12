'''
Module for reporting.

Author: Dauren Baitursyn
Date: 12.09.22
'''
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix

plt.set_loglevel('warning')


with open('config.json', 'r') as f:
    config = json.load(f)


test_data_path = Path.joinpath(Path.cwd(), config['test_data_path'])
model_path = Path.joinpath(Path.cwd(), config['output_model_path'])


def score_model():
    '''Function for reporting.'''

    data = pd.DataFrame()
    target = 'exited'
    for f in os.listdir(test_data_path):
        tmp_df = pd.read_csv(Path.joinpath(test_data_path, f))
        data = data.append(tmp_df)

    y = data.loc[:, target].values.ravel()
    pred = model_predictions(data)

    cm = confusion_matrix(y, pred)
    ax = plt.gca()
    sns.heatmap(cm, cmap="Blues", annot=True, cbar=False, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True")
    plt.savefig(Path.joinpath(model_path, 'confusionmatrix.png'))


if __name__ == '__main__':
    score_model()
