
import json
import numpy as np

from flask import render_template, jsonify
from app import celery

from .model import predict


@celery.task()
def home(data):
    inputs = data['inputs']
    results = predict(inputs)
    result = {'inputs': inputs, 'results': results}
    return render_template('home.html', text=inputs, label=results)


@celery.task()
def main(data):
    inputs = data['inputs']
    results = predict(inputs)
    result = {'inputs': inputs, 'results': results}
    return json.dumps(result)


