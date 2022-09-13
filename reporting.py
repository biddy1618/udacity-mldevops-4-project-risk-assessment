'''
Module for reporting.

Author: Dauren Baitursyn
Date: 12.09.22
'''
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


model_path = Path.joinpath(Path.cwd(), config['output_model_path'])
output_path = Path.joinpath(Path.cwd(), config['output_folder_path'])
test_path = Path.joinpath(Path.cwd(), config['test_data_path'])


def plot_confusion_matrix(production=False, name='confusionmatrix.png'):
    '''Function for plotting confusion matrix.

    Args:
        name (str, optional): Name of the file to save confusion matrix in.
            Defaults to 'confusionmatrix.png'.
    '''
    target = 'exited'
    if production:
        data = pd.read_csv(Path.joinpath(output_path, 'finaldata.csv'))
    else:
        data = pd.read_csv(Path.joinpath(test_path, 'testdata.csv'))

    y = data.loc[:, target].values.ravel()
    pred = model_predictions(data)

    cm = confusion_matrix(y, pred)
    ax = plt.gca()
    sns.heatmap(cm, cmap="Blues", annot=True, cbar=False, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True")
    plt.savefig(Path.joinpath(model_path, name))


if __name__ == '__main__':
    plot_confusion_matrix()
