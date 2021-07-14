from celery import Celery


def make_celery(app_name=__name__):
    celery_app = Celery(app_name)
    default_config = 'app.celery_config'
    celery_app.config_from_object(default_config)

    return celery_app


celery = make_celery()
