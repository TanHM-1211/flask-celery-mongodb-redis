cd /mnt/disk1/tan_hm/sentiment-ja/v2_async/sentiment_ja
# source sentiment_ja_venv/bin/activate
celery multi kill sentiment_ja1 sentiment_ja2 -A celery_worker.celery --pidfile=celery/%n.pid --logfile=log/celery.log 
celery multi start sentiment_ja1 sentiment_ja2 -A celery_worker.celery --loglevel=DEBUG --pidfile=celery/%n.pid --logfile=log/celery.log --pool=solo
#celery worker -A celery_worker.celery  --loglevel=INFO -f log/celery.log --detach --pool=solo
#celery -A celery_worker.celery events
gunicorn -c config/gunicorn_config.py main:app
#python main.py