import logging

from flask import Flask, request
from flask_socketio import SocketIO

logger = logging.getLogger('main')

app = Flask(__name__)
sio_coordinator = SocketIO(app, async_mode='eventlet')


def coordinator_init(addr):
    host = addr.split(':')[0]
    port = addr.split(':')[-1]
    sio_coordinator.run(app, host=host, port=port, debug=False, use_reloader=False)


@sio_coordinator.on('connect')
def connect():
    logger.debug(f'sub_client {request.sid} connected')
    return True  # do auth here


@sio_coordinator.on('disconnect')
def disconnect():
    logger.debug(f'sub client {request.sid} has disconnected')


if __name__ == '__main__':
    coordinator_init('localhost:7000')
