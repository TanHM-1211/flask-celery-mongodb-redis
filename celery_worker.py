from app import celery
from app.factory import create_app, init_celery

app = create_app()
init_celery(celery, app)
