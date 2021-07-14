# docker build -f Dockerfile -t sentiment_ja:v1 .
# FROM ubuntu:18.04
FROM python:3.6.9
USER root
LABEL maintainer="tan.hm1211@gmail.com"
# RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip \
#     && ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip\
#     && apt-get update && apt-get clean all && rm -rf /var/lib/apt/lists/* 
COPY ./requirements.txt /usr
RUN cd /usr  && pip install -r requirements.txt 
COPY . /mnt/disk1/tan_hm/sentiment-ja/v2_async/sentiment_ja/
WORKDIR /mnt/disk1/tan_hm/sentiment-ja/v2_async/sentiment_ja

RUN cd /mnt/disk1/tan_hm/sentiment-ja/v2_async/sentiment_ja && ls \
    && chmod 777 -R ./ && chmod +x -R ./ \
    && mkdir log celery gunicorn

EXPOSE 5001
CMD ["/mnt/disk1/tan_hm/sentiment-ja/v2_async/sentiment_ja/run.sh"]
# ENTRYPOINT ["/bin/bash", "/mnt/disk1/tan_hm/sentiment-ja/v2_async/sentiment_ja/run.sh"]
