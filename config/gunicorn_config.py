loglevel = 'debug'

preload = False
workers = 2
threads = 1
bind = '0.0.0.0:5001'
worker_class = 'gthread'
worker_connections = 2000

pidfile = "log/gunicorn/gunicorn.pid" # pid file
accesslog = "log/gunicorn/access.log" # Access log directory
errorlog = "log/gunicorn/debug.log" # error log
graceful_timeout = 300
timeout = 300

