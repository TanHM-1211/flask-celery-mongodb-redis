from operator import index
import requests
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


# url = 'http://192.168.1.8:2222/test'
# url = 'http://127.0.0.1:1234/explanation'
# url = 'http://sentiment_ja:80/main'
# url = 'http://sa.gpu4hn.hn.aimesoft.com:1880/explanation'
url = 'http://0.0.0.0:5001/sentiment'
base_url = 'http://markitone_sa.demo2.aimenext.com:9080/'
# base_url = 'http://127.0.0.1:5001/'


def test_loop(sentences, n=30):
    res = []
    for sent in sentences:
        res.append({'text': sent, 'result': {'positive': 0, 'negative': 0}, 'score': []})
        params = {'text': sent}

        for i in range(n):
            # r = requests.get(url=url, params=params)
            r = requests.post(url=url, json=params).json()
            res[-1]['result'][r['label']] += 1
            # res[-1]['score'].append(r['score'])
        # res[-1]['score'] = list(np.std(np.array(res[-1]['score']), axis=0))

    return res


def test(sentences):
    res = []
    for sent in sentences:
        params = {'text': sent, 'num_features': 10, 'num_samples': 200}
        r = requests.post(url=base_url, json=params).json()
        res.append(r)

    return res


def test_sentiment(sentences):
    params = {'comments': sentences, 'return_word': 0}
    res = requests.post(url=url, json=params).json()

    return res


def test_create(sentences):
    params = {'comments': sentences, 'return_word': 0}
    res = requests.post(url=base_url + 'asyn-sentiment', json=params).json()

    return res


def test_get_job_id(job_ids=None):
    for job_id in job_ids:
        params = {'id': job_id}

        res = requests.post(url=base_url + 'check-asyn-sentiment', json=params).json()
    return res


def test_get_job_result(job_ids):
    for job_id in job_ids:
        params = {'id': job_id}

        res = requests.post(url=base_url + 'get-asyn-result', json=params).json()
    return res

#
#
sentences = [
    'あまり取り引きしたくない',
    'いつもは本店を利用してます',
    '親切 費用もよくわかりませんが びっくりするほどでない',
    'あんまり良くないから',
    '誠実な対応',
    '料金、スタッフの接客は良かった',
    '料理も美味しかったです',
    '電話対応は人によって違う',
    'フロアスタッフと厨房スタッフが下世話な話をしているのが丸聞こえだった',
    '料理も美味しかったです',
    # 'マリオが子供たちに大人気でした'
]

# params = {'text': sentences[0]}
# r = requests.post(url=base_url+'explanation', json=params).json()
# print(r)

# json.dump(test(sentences), open('./test_res.json', 'w', encoding='utf8'), ensure_ascii=False, indent=4)

# print(test(sentences[:2], n=1))
# for i in test(sentences[:5]):
#     print(i)

print(test_sentiment(['。'.join(sentences[:3]), '。'.join(sentences[3:])]))

# job_id = test_create(sentences)['task_id']
# print(job_id)
# print(test_get_job_id([job_id]))
# print(test_get_job_result([job_id]))

# print(test_get_job_id(
#     sentences=sentences,
#     # job_ids=['604e5505-24f9-4145-90a9-ffa53c3e16b5 ']
# ))

# print(test_get_job_result(
#     job_ids=['f3fec7c9-55f0-4f63-9d25-a6fee4ed2728']
# ))

##########################################


# from .model import predict
# from tqdm import tqdm
# import MeCab
# from lime.lime_text import LimeTextExplainer
# import json

# wakati = MeCab.Tagger("-Owakati")
# def tokenize(text):
#     return wakati.parse(text).split()


# label_name = ['negative', 'positive', 'neutral']
# explainer = LimeTextExplainer(class_names=label_name, split_expression=tokenize)


# def predict_proba(text):
#     probs = []
#     for sentence in tqdm(text):
#         probs.append(list(predict(sentence)[1].values()))
#     probs = np.exp(np.array(probs))
#     return probs / np.sum(probs, axis=1, keepdims=True)

# def get_explanation(text):
#     return explainer.explain_instance(text, predict_proba, num_features=3, num_samples=100,  labels=label_name)

# def test(text):
#     text = text.replace('/', '').replace('\\', '')
#     label, score = predict(text)
#     explanation = []
#     for i, tup in enumerate(get_explanation(text).as_list()):
#         # word = tup[0]
#         # s = tup[1]
#         # label = 'positive'
#         # if s < 0:
#         #     label = 'negative'
#         #     s = -s
#         # explanation.append([word, label, s])
#         explanation.append(tup)

#     # result = {'text': text, 'label': label, 'score': score}
#     result = {'text': text, 'label': label, 'explanation': explanation}

#     return json.dumps(result)

# if __name__ == '__main__':
#     print(test(sentences[1]))


###################################################################
# with open('app/words_polarity.txt', encoding='utf8') as f:
#     sentiment_words = f.read().split('\n')
#     if len(sentiment_words[-1]) < 1:
#         sentiment_words.pop()

# train_df = pd.read_csv('save/real-data-multifit/train.csv')
# neutral = list(train_df['text'][train_df['label'].str.contains('中立')])
# not_neutral = list(train_df['text'][~train_df['label'].str.contains('中立')])

# data = neutral + not_neutral
# labels = [0] * len(neutral) + [1] * len(not_neutral)
# preds = []

# for sentence in data:
#     pred = 0
#     for word in sentiment_words:
#         if word in sentence:
#             pred = 1
#             break
#     preds.append(pred)

# print(classification_report(labels, preds))


# from collections import defaultdict
# res = defaultdict(int)
# set0 = set()
# set1 = set()

# print(len(neutral))
# for i, sentence in enumerate(neutral):
#     flag = False
#     for word in sentiment_words:
#         if word in sentence:
#             flag =  True
#             # res[word] += 1
#             break
#     if not flag:
#         set0.add(i)

# for i, sentence in enumerate(data):
#     flag = False
#     for word in sentiment_words:
#         if word in sentence:
#             flag =  True
#             # res[word] += 1
#             break
#     if not flag:
#         set1.add(i)
# # res = sorted(res.items(), reverse=True, key=lambda x: x[1])
# # print(res[:10])

# mid = set(list(range(len(neutral)))) & set1
# print(len(neutral), len(mid), len(set1))

################################################
# pos_adjs = '美味しい|おいしい|オイシイ|うまい|ウマい|美味い|やさしい|優しい|親切|明るい|気持ち良い|気持ちよい|広い|心地よい|可愛い|好もしい|好き|楽しい|たのしい|ここちよい|程よい|笑顔|便利|充実|偉い|暖かい|あったかい|温かい|美しい|綺麗|きれい|キレイ|新しい|あたらしい|良い|可愛い|かわいい|可愛らしい|見易い|かっこよい|すばらしい|素晴らしい|素敵|すてき|カッコイイ|礼儀正しい|仲良い|清潔|すばやい|上手|すごい|印象深い|分かり易い|嬉しい|感謝|ありがたい|有難い|お世話になりました|ほほえましい|微笑ましい|安い|安全|相応しい|快適|満足|質が良い|近い|丁寧|ていねい'
# neg_adjs = 'キタナイ|汚い|汚れ|きたない|マズイ|まずい|マズい|不味い|怖い|うるさい|煩い|騒々しい|危ない|狭い|古い|気持ち悪い|かゆい|痒い|めんどい|めんどくさい|面倒|邪魔|腹立たしい|つまらない|しんどい|しつこい|臭い|面倒臭い|物足りない|きびしい|厳しい|激しい|せせこましい|堅い|かたい|気味が悪い|さむい|じゃまくさい|失礼|暗い|荒っぽい|痛い|埃っぽい|不便|心配|悩|くだらない|寂しい|悲しい|残念|重い|難しい|つらい|見づらい|おどろおどろしい|ややこしい|遠い|遅い|乏しい|蒸し暑い|騒がしい|眩しい|まぶしい|やかましい|生臭い|不良'

# adjectives = {
#     'positive': [i for i in pos_adjs.split('|')],
#     'negative': [i for i in neg_adjs.split('|')]
# }

# nouns = ['ルーム', 'ホテル', 'レストラン', '部屋']

# y_true = []
# y_pred = []   
# all_text = []    
# for noun in nouns:
#     for tp in adjectives:
#         for adj in adjectives[tp]:
#             text = noun + 'は' + adj + 'です'
#             # print(text)
#             # params = {'text': text}
#             true_label = tp
#             pred_label = requests.get(url='http://markitone_sa.demo2.aimenext.com:9080/main?text={}'.format(text)).json()['label']
#             # print(pred_label)
#             # exit(0)

#             all_text.append(text)
#             y_true.append(true_label)
#             y_pred.append(pred_label)


# print(classification_report(y_true, y_pred))

# t, y0, y1 = [], [], []
# for i in range(len(y_true)):
#     if y_true[i] != y_pred[i]:
#         t.append(all_text[i])
#         y0.append(y_true[i])
#         y1.append(y_pred[i])
# df = pd.DataFrame.from_dict({'text': t, 'true label': y0, 'predict': y1})
# # df = df[df['predict'].str != df['true label'].str]
# df.to_csv('test.csv', index=None)


##############################################
# import MeCab
# mecab_tagger = MeCab.Tagger()


# original_form_pos = 3
# type_pos = 4
# # adj_japan = '形容詞'
# noun_japan = '名詞'

# pos_adjs = '美味しい|おいしい|オイシイ|うまい|ウマい|美味い|やさしい|優しい|親切|明るい|気持ち良い|気持ちよい|広い|心地よい|可愛い|好もしい|好き|楽しい|たのしい|ここちよい|程よい|笑顔|便利|充実|偉い|暖かい|あったかい|温かい|美しい|綺麗|きれい|キレイ|新しい|あたらしい|良い|可愛い|かわいい|可愛らしい|見易い|かっこよい|すばらしい|素晴らしい|素敵|すてき|カッコイイ|礼儀正しい|仲良い|清潔|すばやい|上手|すごい|印象深い|分かり易い|嬉しい|感謝|ありがたい|有難い|お世話になりました|ほほえましい|微笑ましい|安い|安全|相応しい|快適|満足|質が良い|近い|丁寧|ていねい'
# neg_adjs = 'キタナイ|汚い|汚れ|きたない|マズイ|まずい|マズい|不味い|怖い|うるさい|煩い|騒々しい|危ない|狭い|古い|気持ち悪い|かゆい|痒い|めんどい|めんどくさい|面倒|邪魔|腹立たしい|つまらない|しんどい|しつこい|臭い|面倒臭い|物足りない|きびしい|厳しい|激しい|せせこましい|堅い|かたい|気味が悪い|さむい|じゃまくさい|失礼|暗い|荒っぽい|痛い|埃っぽい|不便|心配|悩|くだらない|寂しい|悲しい|残念|重い|難しい|つらい|見づらい|おどろおどろしい|ややこしい|遠い|遅い|乏しい|蒸し暑い|騒がしい|眩しい|まぶしい|やかましい|生臭い|不良'

# adjectives = {
#     'positive': [i for i in pos_adjs.split('|')],
#     'negative': [i for i in neg_adjs.split('|')]
# }

# type_of_adjectives = {i: 'positive' for i in pos_adjs.split('|')}
# type_of_adjectives.update({i: 'negative' for i in neg_adjs.split('|')})


# def mecab_tag(text):
#     """
#     return list of (word, original form, type)
#     """
#     res = []
#     for line in mecab_tagger.parse(text).split('\n')[:-2]:
#         a = line.split('\t')
#         res.append([a[0], a[original_form_pos], a[type_pos].split('-')[0]])
#     return res

# def rule1(text):
#     tagged_text = mecab_tag(text)
#     if len(tagged_text) < 4:
#         return None
    
#     label = None
#     text0 = tagged_text[1][0]
#     text1 = tagged_text[3][0]
#     if tagged_text[0][2] == noun_japan and text0 in ['は', 'が', 'も'] \
#         and text1 in ['たです', 'です', 'だった', 'だ'] and tagged_text[2][1] in type_of_adjectives:
#         label = type_of_adjectives[tagged_text[2][1]]
#     return label 

# text = '部屋は汚いです'

# print(rule1(text))