import json

import celery as cel

from flask import Blueprint, request, render_template, jsonify
from flask_cors import CORS
from celery.result import AsyncResult

from .tasks import sentiment, explanation, main, home, analysis


job_map = {'classify': main, 'sentiment': sentiment, 'explanation': explanation}
status_mapping = {cel.states.FAILURE: 0,
                  cel.states.PENDING: 1,
                  cel.states.RECEIVED: 2,
                  cel.states.RETRY: 3,
                  cel.states.REVOKED: 4,
                  cel.states.STARTED: 5,
                  cel.states.SUCCESS: 6}


bp = Blueprint('all', __name__)
cors = CORS(bp)


@bp.route('/', methods=['GET', 'POST'])
def _home():
    data = {}
    if 'textbox' in request.form:
        data['input'] = str(request.form.get('textbox'))
    else:
        return render_template('home.html')
    return home(data)


@bp.route('/main', methods=['GET', 'POST'])
def _main():
    data = {}
    if request.method == 'GET' and 'text' in request.args:
        data = dict(request.args)
        data['text'] = str(request.args['text'])
        return main(data)
    elif request.method == 'POST':
        data = request.get_json()
        data['text'] = request.get_json()['text']
    else:
        return None
    return main.delay((data)).get()


@bp.route('/analysis', methods=['GET', 'POST'])
def _analysis():
    data = {}
    if request.method == 'GET' and 'text' in request.args:
        data = dict(request.args)
        data['text'] = str(request.args['text'])
        return analysis(data)
    elif request.method == 'POST':
        data = request.get_json()
        data['text'] = request.get_json()['text']
    else:
        return None
    return analysis.delay((data)).get()


@bp.route('/explanation', methods=['GET', 'POST'])
def _explanation():
    data = {}
    if request.method == 'GET' and 'text' in request.args:
        data['text'] = str(request.args['text'])
        if 'num_features' in request.args:
            data['explanation_num_features'] = int(request.args['num_features'])
        if 'num_samples' in request.args:
            data['explanation_num_samples'] = int(request.args['num_samples'])
        return explanation(data)
    elif request.method == 'POST':
        json_data = request.get_json()
        data['text'] = json_data['text']
        if 'num_features' in data:
            data['explanation_num_features'] = json_data['num_features']
        if 'num_samples' in data:
            data['explanation_num_samples'] = json_data['num_samples']
    else:
        return None
    return explanation.delay((data)).get()

    # return {'text': text}


@bp.route('/sentiment', methods=['POST'])
def _sentiment():
    if request.method != 'POST':
        result = {
            'status': 'failed',
            'message': 'not a post request',
            'result': None,
        }
        return result
    data = request.get_json()
    return sentiment.delay((data)).get()


@bp.route('/asyn-sentiment', methods=['POST'])
def create_job():
    result = {"task_id": None, 'message': None, 'status': None}
    try:
        params = request.get_json()
        data = params

        job = sentiment.delay((data))
        result['task_id'] = job.task_id
        result['status'] = cel.states.SUCCESS
    except Exception as e:
        result['status'] = cel.states.FAILURE
        result['message'] = str(e)
    return jsonify(result)


from app import celery
@bp.route('/check-asyn-sentiment', methods=['GET', 'POST'])
def get_status():
    result = {"result": None, 'message': None, 'status': None}
    try:
        if request.method == 'GET':
            job_id = str(request.args['id'])
        else:
            params = request.get_json()
            job_id = params['id']
        job = AsyncResult(job_id, app=celery)
        if job.state == cel.states.FAILURE or job.state == cel.states.REVOKED:
            return get_result()
        else:
            result['status'] = cel.states.SUCCESS
            result['result'] = job.state
    except Exception as e:
        result['status'] = cel.states.FAILURE
        result['message'] = str(e)
    return jsonify(result)


@bp.route('/get-asyn-result', methods=['GET', 'POST'])
def get_result():
    result = {"result": None, 'message': None,
              'status': None}
    try:
        if request.method == 'GET':
            job_id = str(request.args['id'])
        else:
            params = request.get_json()
            job_id = params['id']
        job_result = json.loads(AsyncResult(job_id, app=celery).get(propagate=True))
        result['result'] = job_result['result']
        result['message'] = job_result['message']
        result['status'] = job_result['status']
    except Exception as e:
        result['status'] = cel.states.FAILURE
        result['message'] = str(e)
    return jsonify(result)


