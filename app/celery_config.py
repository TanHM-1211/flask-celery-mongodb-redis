## Broker settings.
broker_url = 'redis://127.0.0.1:6378/'

result_backend = 'mongodb://localhost:27017/'
mongodb_backend_settings = {
    'database': 'test_db',
    'taskmeta_collection': 'save',
}

result_expires = None
result_extended = True
task_track_started = True
task_default_queue = 'test_queue'