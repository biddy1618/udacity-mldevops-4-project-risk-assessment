'''
Script for calling API endpoints and saving the content
to file 'apireturns.txt'.

Author: Dauren Baitursyn
Date: 12.09.22
'''
import requests
from pathlib import Path

URL = 'http://127.0.0.1'
PORT = '8000'


response1 = requests.post(
    f'{URL}:{PORT}/prediction?dataset=testdata/testdata.csv').text
response2 = requests.get(f'{URL}:{PORT}/scoring').text
response3 = requests.get(f'{URL}:{PORT}/summarystats').text
response4 = requests.get(f'{URL}:{PORT}/diagnostics').text

responses = 'prediction:\n' + response1
responses += '\nscoring:\n' + response2
responses += '\nsummarystats:\n' + response3
responses += '\ndiagnostics:\n' + response4

with open(Path.joinpath(Path.cwd(), 'apireturns.txt'), 'w') as f:
    f.write(responses)
