from flask import Flask

PKG_NAME = 'app'


def init_celery(celery, app):
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                # return celery.Task.__call__(self, *args, **kwargs)
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    celery.finalize()


def create_app(app_name=PKG_NAME, **kwargs):
    app = Flask(app_name)
    if kwargs.get('celery'):
        init_celery(kwargs.get('celery'), app)
    from app.main import bp
    app.register_blueprint(bp)
    return app
