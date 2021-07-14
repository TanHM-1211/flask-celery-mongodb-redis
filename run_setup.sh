#!/bin/bash
virtualenv -p /usr/bin/python3.6 ./sentiment_ja_venv
ls
source ./sentiment_ja_venv/bin/activate
pip show torch
pip install -r requirements.txt --use-deprecated=legacy-resolver
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1p7Apqqv9CMFYZC9UQfVbTDcUc2avFOsV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1p7Apqqv9CMFYZC9UQfVbTDcUc2avFOsV" -O all_data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14IwZzXFq3lxdV9oGMz18xReIwKibpD6p' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14IwZzXFq3lxdV9oGMz18xReIwKibpD6p" -O save.zip && rm -rf /tmp/cookies.txt
mkdir save
#unzip all_data.zip -d save
unzip save.zip
cd multifit
python setup.py install
cd ..
mkdir gunicorn ; mkdir log
cd mongodb ; mkdir mongodb_data ; sudo chown 1001 ./mongodb_data ; sudo docker-compose up -d ; cd ..
cd redis ; mkdir redis_data ; sudo chown 1001 ./redis_data ; sudo docker-compose up -d ; cd ..
sudo docker update --restart unless-stopped mongodb_sentiment_ja
sudo docker update --restart unless-stopped redis_sentiment_ja
