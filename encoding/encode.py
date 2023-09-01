import argparse
import json

import numpy as np
import pandas as pd
import torch

from data_prep import data_prep
from dataset import split
from model import get_model
from utils import cal_int_encoding, df_to_str, cal_threshold


def run():
    print('Preparing dataset for selecting the threshold:')
    data_l, data_r, data_match = data_prep(args.data_name)
    tr_data_in_split, _, _, train_dataset, val_dataset, test_dataset = \
        split(data_l, data_r, data_match, 0.6, 0.2, 0.2, attribute=args.attr, tr_ratio=args.ratio, ret_labels=False, verbose=False)
    train_data_l, train_data_r, tr_labels, tr_match_ind, tr_non_match_ind, tr_data_l_str, tr_data_r_str = tr_data_in_split
    th_select_data = tr_data_l_str, tr_data_r_str, tr_match_ind, tr_non_match_ind

    print(f'Loading checkpoint: {args.ckpt_path}')
    ckpt = torch.load(args.ckpt_path)
    # get the encoding_size from the model ckpt (the length of the last (linear) layer)
    encoding_size = len(list(ckpt.items())[-1][1])
    model = get_model(args.model_name, encoding_size=encoding_size).to(device)
    model.load_state_dict(ckpt)

    print('Calculating threshold:')
    threshold, f_score, max_val, min_val = cal_threshold(model, th_select_data, test_bs=args.bs,
                                                         metric='hamming', num_interval=args.num_interval, verbose=False)
    if threshold >= encoding_size // 2:
        print(f'Calculated threshold {threshold} >= {encoding_size // 2}, use {encoding_size // 2} instead.')
    else:
        print(f'Threshold: {threshold}, f_score: {f_score}, max_val: {max_val}, min_val: {min_val}')

    print('Calculate and save encodings')
    df = pd.read_csv(args.data_path, encoding='unicode_escape', index_col=False)
    data_str = df_to_str(df, args.attr)  # convert data records to strings
    encoding = cal_int_encoding(model, data_str, test_bs=args.bs, max_val=max_val, min_val=min_val, num_interval=args.num_interval)
    np.save(args.save_path, encoding)


def args_parser():
    parser = argparse.ArgumentParser()

    # command line arguments
    parser.add_argument('--load_config', type=str, default=None, help='the config file')

    parser.add_argument('-d', '--data_path', type=str, help='path of the data file; expect a .csv file')
    parser.add_argument('--data_name', type=str, help='dataset name', choices=['ag', 'febrl4', 'abt_buy'])
    parser.add_argument('-n', '--model_name', type=str, default='all-distilroberta-v1',
                        help='model architecture name for inference')
    parser.add_argument('-p', '--ckpt_path', type=str, help='path of the model checkpoint')
    parser.add_argument('-i', '--num_interval', type=int, default=8,
                        help='number of intervals for quantization (converting embeddings to integer vectors)')
    parser.add_argument('-r', '--ratio', type=int, default=1000,
                        help='ratio (# of negatives /# of positives) of the dataset for selecting the threshold')

    parser.add_argument('--bs', type=int, default=512, help='batch size used during inference')

    parser.add_argument('--attr', type=str, default=None, help='attribute to extract from the csv; None for using all ')

    parser.add_argument('-s', '--save_path', type=str, help='path of the encodings output')

    args = parser.parse_args()

    # load json file
    if args.load_config is not None:
        with open(args.load_config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    return args


if __name__ == "__main__":
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run()
    print('end')
