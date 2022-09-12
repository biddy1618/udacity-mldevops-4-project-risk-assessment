'''
Module for API setup.

Author: Dauren Baitursyn
Date: 12.09.22
'''
from flask import Flask, Response, request, jsonify
# from flask import session
from pathlib import Path
import pandas as pd
import json

import diagnostics
from scoring import score_model

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

model_path = Path.joinpath(Path.cwd(), config['output_model_path'])
dataset_csv_path = Path.joinpath(
    Path.cwd(), config['output_folder_path'])


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    '''
    Endpoint for prediction for the given dataset.
    Expects 'dataset' query string as the location of the dataset.


    Returns:
        list: List of predictions
    '''
    dataset = request.args.get('dataset')
    data = pd.read_csv(Path.joinpath(Path.cwd(), dataset))
    pred = diagnostics.model_predictions(data)
    return jsonify({'predictions': pred})


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    '''
    Endpoint for scoring.

    Returns:
        float: F1 score of deployed model.
    '''
    f1_score = score_model()

    return jsonify({'F1 score': f1_score})


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary_stats():
    '''
    Endpoint for running summary statistics.

    Returns:
        list[list]: List of statistics in order means, medians,
            and standard deviations.
    '''
    data = pd.read_csv(Path.joinpath(dataset_csv_path, 'finaldata.csv'))
    stats = diagnostics.dataframe_summary(data)

    return jsonify({
        'Means': stats[0],
        'Medians': stats[1],
        'Standard Deviation': stats[2]
    })


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnose():
    '''
    Endpoint for checking timing and percent NA values.

    Returns:
        str: Diagnostics stats on timing, missing data, and outdated packages.
    '''
    data = pd.read_csv(Path.joinpath(dataset_csv_path, 'finaldata.csv'))
    missing = diagnostics.missing_data(data)
    exec_time = diagnostics.execution_time()
    out_packages = diagnostics.outdated_packages_list()
    response_text = 'Missing data:\n' + str(missing) + \
        '\nExecution time:\n' + str(exec_time) + \
        '\nOutdated packages:\n' + str(out_packages)

    return Response(response_text, mimetype='text/plain')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
