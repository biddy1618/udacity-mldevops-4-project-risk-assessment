# Project details

Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of your work, though. Your industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, you need to set up regular monitoring of your model to ensure that it remains accurate and up-to-date. You'll set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.

## Project folders

Your workspace has seven locations you should be aware of:

* `/practicedata/`. This is a directory that contains some data you can use for practice.
* `/sourcedata/`. This is a directory that contains data that you'll load to train your models.
* `/ingesteddata/`. This is a directory that will contain the compiled datasets after your ingestion script.
* `/testdata/`. This directory contains data you can use for testing your models.
* `/models/`. This is a directory that will contain ML models that you create for production.
* `/practicemodels/`. This is a directory that will contain ML models that you create as practice.
* `/production_deployment/`. This is a directory that will contain your final, deployed models.

## Project files

The following are the Python files that are in the starter files:

* `training.py`, a Python script meant to train an ML model
* `scoring.py`, a Python script meant to score an ML model
* `deployment.py`, a Python script meant to deploy a trained ML model
* `ingestion.py`, a Python script meant to ingest new data
* `diagnostics.py`, a Python script meant to measure model and data diagnostics
* `reporting.py`, a Python script meant to generate reports about model metrics
* `app.py`, a Python script meant to contain API endpoints
* `wsgi.py`, a Python script to help with API deployment
* `apicalls.py`, a Python script meant to call your API endpoints
* `fullprocess.py`, a script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed

## Datasets

The following are the datasets that are included in your starter files. Each of them is fabricated datasets that have information about hypothetical corporations.

Note: these data have been uploaded to your workspace as well

* `dataset1.csv` and `dataset2.csv`, found in `/practicedata/`
* `dataset3.csv` and `dataset4.csv`, found in `/sourcedata/`
* `testdata.csv`, found in `/testdata/`

## Other files

The following are other files that are included in your starter files:

* `requirements.txt`, a text file and records the current versions of all the modules that your scripts use
* `config.json`, a data file that contains names of files that will be used for configuration of your ML Python scripts

## Notes

For the first run: `ingestion.py` -> `training.py` -> `scoring.py` -> `deployment.py` -> `diagnostics.py` -> `app.py` -> `reporting.py`.

Order of execution:
1) `ingestion.py` - composes new file at `output_folder_path` - `finaldata.csv` and `ingestedfiles.txt`.
2) `scoring.py` - takes model at `prod_deployment_path` - `trainedmodel.pkl`, and scores data at `output_folder_path` - `finaldata.csv`, and saves the score at `output_model_path` - `latestscore.txt`.
3) `training.py` - trains new model for the data at `output_folder_path` - `finaldata.csv` and saves it at `output_model_path` - `trainedmodel.pkl`.
4) `deployment.py` - copies model and score from `output_model_path` - `trainedmodel.pkl` and `latestscore.txt` - and list of ingested files at `output_folder_path` - `ingestedfiles.txt` to deployment folder `prod_deployment_path`.
5) `reporting.py` - composes confusion matrix from production model at `prod_deployment_path` for the file at `output_folder_path` and saves it in `output_model_path`.
6) `app.py` - starts Flask API for the model.
7) `apicall.py` - generates output file for by calling API endpoints.