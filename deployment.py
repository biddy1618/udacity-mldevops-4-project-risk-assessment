'''
Module for deploying model with latest score and ingested files.

Author: Dauren Baitursyn
Date: 11.09.22
'''
import json
import shutil

from pathlib import Path


with open('config.json', 'r') as f:
    config = json.load(f)

prod_path = Path.joinpath(Path.cwd(), config['prod_deployment_path'])
model_path = Path.joinpath(Path.cwd(), config['output_model_path'])
output_path = Path.joinpath(Path.cwd(), config['output_folder_path'])


def deploy_model():
    '''Function for deployment.'''

    # copy the model
    shutil.copyfile(
        Path.joinpath(model_path, 'trainedmodel.pkl'),
        Path.joinpath(prod_path, 'trainedmodel.pkl')
    )

    # copy the score
    shutil.copyfile(
        Path.joinpath(model_path, 'latestscore.txt'),
        Path.joinpath(prod_path, 'latestscore.txt')
    )

    # copy the ingested files file
    shutil.copyfile(
        Path.joinpath(output_path, 'ingestedfiles.txt'),
        Path.joinpath(prod_path, 'ingestedfiles.txt')
    )


if __name__ == '__main__':
    deploy_model()
