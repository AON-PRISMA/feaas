import logging
import sys


def get_logger(debug, name=None):
    log_level = logging.DEBUG if debug else logging.INFO
    if name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
