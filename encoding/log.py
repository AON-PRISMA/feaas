import json
import logging
import os
import sys


def init(level, file_path=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    log_dir = None
    if file_path is not None:
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        log_dir = get_log_dir(file_path)
        handler = logging.FileHandler(filename=os.path.join(log_dir, 'output.log'), mode="a")
        handlers.append(handler)
    logging.basicConfig(level=level, handlers=handlers,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', force=True)
    return log_dir


def reset_basic_config(level, file_path):
    handlers = [logging.StreamHandler(sys.stdout)]
    if file_path is not None:
        # search for the latest dir:
        i = 1
        while os.path.exists(os.path.join(file_path, f'run_{i}')):
            i += 1
        handler = logging.FileHandler(filename=os.path.join(os.path.join(file_path, f'run_{i-1}'), 'output.log'), mode="a")
        handlers.append(handler)
    logging.basicConfig(level=level, handlers=handlers,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', force=True)


def get_log_dir(file_path):
    i = 1
    while os.path.exists(os.path.join(file_path, f'run_{i}')):
        i += 1
    os.mkdir(os.path.join(file_path, f'run_{i}'))
    return os.path.join(file_path, f'run_{i}')


def get_logger(name=None):
    if name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(name)
    return logger


def search_configs(args, file_path):
    train_params = ['dataset_name', 'model_name', 'encoding_size', 'lr', 'epochs', 'bs', ' warmup_prop', 'temperature',
                    'tr_dist', 'attr', 'mode', 'tr_ratio']
    args_dict = vars(args)
    args_dict = {k: v for k, v in args_dict.items() if k in train_params}  # consider only training_relevant params

    matched_i = []
    i = 1
    while os.path.exists(f'{file_path}/run_{i}'):
        if not os.path.exists(f'{file_path}/run_{i}/config.json'):
            continue
        with open(f'{file_path}/run_{i}/config.json', 'r') as f:
            configs = json.load(f)
        configs = {k: v for k, v in configs.items() if k in train_params}
        if args_dict == configs:
            print(f'Found matched configs: {i}')
            matched_i.append(i)
        i += 1
    if len(matched_i) == 0:
        return ''
    # return f'{file_path}/run_{matched_i[-1]}/model.pth'  # return the last matched path
    return os.path.join(file_path, f'run_{matched_i[-1]}', 'model.pth')
