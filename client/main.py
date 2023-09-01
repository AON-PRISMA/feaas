import argparse
import json

import log
from client import Client
from dataset import get_data, get_test_data


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config', type=str, default=None, help='the config file')

    parser.add_argument('--data_path', type=str, help='expected a .npy file')

    parser.add_argument('--cid', type=int, default=-1, help='client id')

    parser.add_argument('--server_addr', type=str, default='127.0.0.1:8000', help='ip address to connect to the server')
    parser.add_argument('--client_addr', type=str, default='127.0.0.1:7000', help='ip address used between clients')
    parser.add_argument('--tls_file', type=str, default='', help='path to the tls file for verification; '
                                                                 'if not provided, non-tls is used')
    parser.add_argument('--test_data', type=int, default=0, help='whether to use test data or not')

    parser.add_argument('--th', type=int, help='the decision making threshold')
    parser.add_argument('--rep', type=int, default=1, help='number of repetitions for encryption to improve security')
    parser.add_argument('--lsh_rep', type=int, default=10, help='repetition param for lsh; non-positive means no lsh')
    parser.add_argument('--lsh_num_ind', type=int, default=7, help='number of selected indices for lsh')
    parser.add_argument('--lsh_num_bin', type=int, default=16, help='number of bins for lsh')

    parser.add_argument('--num_bits', type=int, default=32, help='number of bits of the protocol')

    parser.add_argument('--plain', type=int, default=0, help='send plain texts directly or not. For testing purposes')
    parser.add_argument('--debug', type=int, default=0, help='whether to enable debug mode')

    args = parser.parse_args()

    # load json file
    if args.load_config is not None:
        with open(args.load_config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    return args


def main():
    args = args_parser()
    logger = log.get_logger(args.debug, name='main')
    if args.test_data:
        record = get_test_data(args)
    else:
        record = get_data(args)

    logger.info('Client starts:')
    cur_client = Client(record, args)
    cur_client.run()
    logger.info('Done')


main()
