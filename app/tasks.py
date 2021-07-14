import datetime
import json
import numpy as np
import MeCab
import time

from flask import render_template, jsonify
from lime.lime_text import LimeTextExplainer
from app import celery

from .config import use_neutral
from .model import predict
# def predict(text):
#     # time.sleep(1)
#     return 'positive', [1, 0, 0]


wakati = MeCab.Tagger("-Owakati")
punct = set(list('！ ” ＃ ＄ ％ ＆ ’ （ ） ＊ ＋ 、 − ． ／ ： ； ＜ ＝ ＞ ？ ＠ ＼ ＾ ＿ ｀ ｜ 〜[,.、。!・？?「」"~() ') + ["'"])
with open('app/stop_words.txt', encoding='utf8') as f:
    stop_words = f.read().split()
    if (stop_words[-1]) == '':
        stop_words.pop()


# Lime
def tokenize(text):
    return [i for i in wakati.parse(text).split() if i not in stop_words]


# label_name = ['negative', 'positive', 'neutral']
# if not use_neutral:
#     label_name = ['negative', 'positive']
label_name = ['negative', 'positive']

explainer = LimeTextExplainer(class_names=label_name, split_expression=tokenize)


def predict_proba(text):
    probs = []
    for sentence in text:
        probs.append(list(predict(sentence, get_raw_score=True)[1].values()))
    probs = np.exp(np.array(probs))
    return probs / np.sum(probs, axis=1, keepdims=True)


def get_explanation(text, num_features=3, num_samples=100):
    return explainer.explain_instance(text, predict_proba, num_features=num_features,
                                      num_samples=num_samples)


@celery.task()
def home(data):
    text = data['text']
    text = text.replace('/', '').replace('\\', '')

    label, score = predict(text)
    # result = {'text': text, 'label': label}
    return render_template('home.html', text=text, label=label)


@celery.task()
def main(data):
    text = data['text']
    text = text.replace('/', '').replace('\\', '')
    label, score = predict(text)
    result = {'text': text, 'label': label}
    return json.dumps(result)


@celery.task()
def analysis(data):
    text = data['text']
    get_score = 'True'
    if 'get_score' in data:
        get_score = data['get_score']

    text = text.replace('/', '').replace('\\', '')

    if get_score == 'True' and use_neutral:
        label, score, hotel_review_model_score, real_data_model_score = predict(text, get_all_score=True, normalize_before_add=True)
        result = {'text': text, 'label': label, 'score': score,
                   'hotel_review_model_score': hotel_review_model_score, 'real_data_model_score': real_data_model_score }
    else:
        label, score = predict(text)
        result = {'text': text, 'label': label, 'score': score}
    return json.dumps(result)


@celery.task()
def explanation(data):
    result = {
        'status': 'success',
        'message': None,
        'result': {},
    }
    try:
        explanation_num_features = 5
        explanation_num_samples = 200

        text = data['text']
        if 'num_features' in data:
            explanation_num_features = data['num_features']
        if 'num_samples' in data:
            explanation_num_samples = data['num_samples']

        text = text.replace('/', '').replace('\\', '')
        label, score = predict(text)
        explanation = {}

        for i, tup in enumerate(get_explanation(text, explanation_num_features, explanation_num_samples).as_list()):
            word = tup[0]
            s = tup[1]
            label_ = 'negative'
            if s < 0:
                label_ = 'positive'
                s = -s
            explanation[i+1] = [word, label_, s]

        # result = {'text': text, 'label': label, 'score': score}
        result['result'] = {'text': text, 'label': label, 'score': score, 'explanation': explanation}
    except Exception as e:
        result = {
            'status': 'failed',
            'message': str(e),
            'result': None,
        }
    return json.dumps(result, ensure_ascii=False)

    # return {'text': text}


@celery.task()
def sentiment(data):
    t1 = datetime.datetime.now()
    result = {
        'status': 'success',
        'message': None,
        'result': {},
    }

    def make_new_word(word, all_words):
        if use_neutral:
            all_words[word] = {'all frequency': 0, "frequency": {"positive": 0, "negative": 0, "neutral": 0}, 'comment_inds': []}
        else:
            all_words[word] = {'all frequency': 0, "frequency": {"positive": 0, "negative": 0}, 'comment_inds': []}
    try:
        result['result'] = {'sentiment': [], 'words': []}
        all_words = {}  # w: [all_freq, [pos_freq, neg_freq, neu_freq], comment_ids]
        comments = data['comments']
        score_for_label = {'positive': 1, 'neutral': 0, 'negative': -1}
        return_word = True
        if 'return_word' in data:
            return_word = True if int(data['return_word']) == 1 else False

        split_token = '。'
        for cmt_id, comment in enumerate(comments):
            sentences = [i for i in comment.split(split_token) if len(i) > 1]
            result['result']['sentiment'].append([])
            for sentence in sentences:
                label, score = predict(sentence + split_token)
                result['result']['sentiment'][-1].append({"sen": sentence, 'label': label, 'score': score})

                if return_word:
                    words = tokenize(sentence)
                    for word in words:
                        if word not in all_words:
                            make_new_word(word, all_words)
                        all_words[word]['all frequency'] += 1
                        all_words[word]['frequency'][label] += 1
                        all_words[word]['comment_inds'].append(cmt_id)
            result['result']['sentiment'][-1] = {"sens": result['result']['sentiment'][-1]}
            overall_label = 0
            for sentence in result['result']['sentiment'][-1]['sens']:
                overall_label += score_for_label[sentence['label']]
            overall_label = min(max(overall_label, -1), 1)
            for k in score_for_label:
                if overall_label == score_for_label[k]:
                    overall_label = k
                    break
            result['result']['sentiment'][-1]['overall_label'] = overall_label
            
        if return_word:
            all_words = dict(sorted(list(all_words.items()), key=lambda x: x[1]['all frequency'], reverse=True))
            for word in all_words:
                result['result']['words'].append({'surface': word, 'frequency': all_words[word]['frequency'],
                                              'comment_inds': all_words[word]['comment_inds']})

    except Exception as e:
        result = {
            'status': 'FAILED',
            'message': str(e),
            'result': None,
        }

    t2 = datetime.datetime.now()
    result['exe_time'] = (t2 - t1).total_seconds()
    return json.dumps(result, ensure_ascii=False)


