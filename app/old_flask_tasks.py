import json
import numpy as np
import MeCab

from flask import jsonify, render_template

from .model import predict
from lime.lime_text import LimeTextExplainer


wakati = MeCab.Tagger("-Owakati")


punct = set(list('！ ” ＃ ＄ ％ ＆ ’ （ ） ＊ ＋ 、 − ． ／ ： ； ＜ ＝ ＞ ？ ＠ ＼ ＾ ＿ ｀ ｜ 〜[,.、。!・？?「」"~() ') + ["'"])


# Lime
def tokenize(text):
    return [i for i in wakati.parse(text).split() if i not in punct]


explainer = LimeTextExplainer(class_names=['negative', 'positive'], split_expression=tokenize)


def predict_proba(text):
    probs = []
    for sentence in text:
        probs.append(list(predict(sentence)[1].values()))
    probs = np.exp(np.array(probs))
    return probs / np.sum(probs, axis=1, keepdims=True)


def get_explanation(text, num_features=3, num_samples=100):
    return explainer.explain_instance(text, predict_proba, num_features=num_features,
                                      num_samples=num_samples)


def home(request):
    if 'textbox' in request.form:
        text = str(request.form.get('textbox'))
    else:
        return render_template('home.html')
    text = text.replace('/', '').replace('\\', '')

    label, score = predict(text)
    # result = {'text': text, 'label': label}
    return render_template('home.html', text=text, label=label)


def main(request):
    if request.method == 'GET' and 'text' in request.args:
        text = str(request.args['text'])
    elif request.method == 'POST':
        data = request.get_json()
        text = data['text']
    else:
        return None
    text = text.replace('/', '').replace('\\', '')
    label, score = predict(text)

    # result = {'text': text, 'label': label, 'score': score}
    result = {'text': text, 'label': label}
    return jsonify(result)

    # return {'text': text}


def explanation(request):
    explanation_num_features = 5
    explanation_num_samples = 200
    if request.method == 'GET' and 'text' in request.args:
        text = str(request.args['text'])
        if 'num_features' in request.args:
            explanation_num_features = int(request.args['num_features'])
        if 'num_samples' in request.args:
            explanation_num_samples = int(request.args['num_samples'])
    elif request.method == 'POST':
        data = request.get_json()
        text = data['text']
        if 'num_features' in data:
            explanation_num_features = data['num_features']
        if 'num_samples' in data:
            explanation_num_samples = data['num_samples']
    else:
        return None
    text = text.replace('/', '').replace('\\', '')
    label, score = predict(text)
    explanation = {}

    for i, tup in enumerate(get_explanation(text, explanation_num_features, explanation_num_samples).as_list()):
        word = tup[0]
        s = tup[1]
        label = 'positive'
        if s < 0:
            label = 'negative'
            s = -s
        explanation[i+1] = [word, label, s]

    # result = {'text': text, 'label': label, 'score': score}
    result = {'text': text, 'label': label, 'score': score, 'explanation': explanation}
    return json.dumps(result)

    # return {'text': text}


def sentiment(request):
    result = {
        'status': 'success',
        'message': None,
        'result': {},
    }
    if request.method != 'POST':
        result = {
            'status': 'failed',
            'message': 'not a post request',
            'result': None,
        }
        return result
    else:
        def make_new_word(word, all_words):
            all_words[word] = {'all frequency': 0, "frequency": {"positive": 0, "negative": 0, "neutral": 0}, 'comment_inds': []}
        try:
            result['result'] = {'sentiment': [], 'words': []}
            all_words = {}  # w: [all_freq, [pos_freq, neg_freq, neu_freq], comment_ids]
            data = request.get_json()
            comments = data['comments']
            for cmt_id, comment in enumerate(comments):
                sentences = [i for i in comment.split('。') if len(i) > 1]
                result['result']['sentiment'].append([])
                for sentence in sentences:
                    label, score = predict(sentence)
                    result['result']['sentiment'][-1].append({"sen": sentence, 'label': label})

                    words = tokenize(sentence)
                    for word in words:
                        if word not in all_words:
                            make_new_word(word, all_words)
                        all_words[word]['all frequency'] += 1
                        all_words[word]['frequency'][label] += 1
                        all_words[word]['comment_inds'].append(cmt_id)

            all_words = dict(sorted(list(all_words.items()), key=lambda x: x[1]['all frequency'], reverse=True))
            for word in all_words:
                result['result']['words'].append({'surface': word, 'frequency': all_words[word]['frequency'],
                                                  'comment_inds': all_words[word]['comment_inds']})

        except Exception as e:
            result = {
                'status': 'failed',
                'message': str(e),
                'result': None,
            }
    return json.dumps(result)
