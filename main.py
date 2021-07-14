from app import factory
import app
import logging

logger = logging.getLogger('werkzeug')  # grabs underlying WSGI logger
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/flask.log')  # creates handler for the log file
logger.addHandler(handler)

app = factory.create_app(celery=app.celery)

if __name__ == '__main__':

    app.run(host='127.0.0.1', port='5001')