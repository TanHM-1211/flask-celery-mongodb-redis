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
url = 'http://0.0.0.0:5001/main'


def test(sentences):
    res = []
    for sent in sentences:
        params = {'input': [sent]}
        r = requests.post(url=url, json=params).json()
        res.append(r)

    return res

