import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = log.get_logger(__name__)


def convert_to_str(row, attribute):
    if attribute is None:
        value = row.to_dict()
        value_str = json.dumps(value)
        return value_str
    else:
        return row[attribute]


def df_to_str(df, attribute):
    data_str = []
    for _, row in df.iterrows():
        data_str.append(convert_to_str(row, attribute))
    return data_str


def cal_int_encoding(model, dataset, test_bs, max_val, min_val, num_interval):
    encode = model.encode(dataset, batch_size=test_bs, convert_to_numpy=False, show_progress_bar=True)
    if max_val is None and min_val is None:
        print('Auto calculating max_val and min_val...')
        max_val = torch.max(torch.stack(encode)).item()
        min_val = torch.min(torch.stack(encode)).item()
        print(f'max_val: {max_val}, min_val: {min_val}')
    encode = convert_to_int(encode, max_val, min_val, num_interval)
    return encode


def convert_to_int(to_encode, max_val, min_val, num_interval=8):  # converts real vectors to integer vectors
    with torch.no_grad():
        to_encode = torch.stack(to_encode).cpu()
        bound = np.linspace(min_val, max_val, num_interval, endpoint=False)[1:]  # num_interval-1 bounds
        bound = torch.from_numpy(bound)
        bucket_encodings = torch.bucketize(to_encode, bound)
        ret = bucket_encodings.numpy()
        return ret


def cal_encoding(model, dataset, test_bs, multi_gpu):
    data_l, data_r, match_ind, non_match_ind = dataset
    whole_str_list = data_l + data_r  # combine the string lists
    if multi_gpu:
        workers = model.start_multi_process_pool(None)  # use all available gpus
        encodings = model.encode_multi_process(whole_str_list, pool=workers, batch_size=test_bs)
        SentenceTransformer.stop_multi_process_pool(workers)
        encodings = torch.from_numpy(encodings).to(device)  # convert to tensors
        encodings = list(encodings)
    else:
        encodings = model.encode(whole_str_list, batch_size=test_bs, convert_to_numpy=False, show_progress_bar=True)

    encode_l = encodings[:len(data_l)]
    encode_r = encodings[len(data_l):]

    # select the encodings that are sampled in the training set
    whole_ind_list = match_ind + non_match_ind
    index_list_1 = [i for i, _ in whole_ind_list]
    index_list_2 = [j for _, j in whole_ind_list]
    encode_1 = [encode_l[i] for i in index_list_1]
    encode_2 = [encode_r[j] for j in index_list_2]
    labels = [1] * len(match_ind) + [0] * len(non_match_ind)
    return encode_1, encode_2, labels


def cal_dist(model, data, test_bs, metric='hamming', max_val=None, min_val=None, num_interval=None, multi_gpu=False):
    encode_1, encode_2, labels = cal_encoding(model, data, test_bs, multi_gpu)
    if metric == 'hamming':
        if max_val is None and min_val is None:
            max_val = max(torch.max(torch.stack(encode_1)).item(), torch.max(torch.stack(encode_2)).item())
            min_val = min(torch.min(torch.stack(encode_1)).item(), torch.min(torch.stack(encode_2)).item())

        encode_1 = convert_to_int(encode_1, max_val, min_val, num_interval)
        encode_2 = convert_to_int(encode_2, max_val, min_val, num_interval)

        dist = np.sum(encode_1 != encode_2, axis=1)
    elif metric == 'l1':
        if max_val is None and min_val is None:
            max_val = max(torch.max(torch.stack(encode_1)).item(), torch.max(torch.stack(encode_2)).item())
            min_val = min(torch.min(torch.stack(encode_1)).item(), torch.min(torch.stack(encode_2)).item())

        encode_1 = convert_to_int(encode_1, max_val, min_val, num_interval)
        encode_2 = convert_to_int(encode_2, max_val, min_val, num_interval)

        dist = np.linalg.norm(encode_1 - encode_2, ord=1, axis=1)
    else:
        raise NotImplementedError
    return encode_1, encode_2, labels, dist, max_val, min_val


def cal_stat(dist, th, labels):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(labels)):
        if dist[i] < th and labels[i] == 1:
            tp += 1
        elif dist[i] < th and labels[i] == 0:
            fp += 1
        elif dist[i] >= th and labels[i] == 1:
            fn += 1
        elif dist[i] >= th and labels[i] == 0:
            tn += 1
    return tp, tn, fp, fn


def cal_threshold(model, tr_data_in_split, test_bs, metric='hamming', num_interval=8, multi_gpu=False, verbose=True):
    encode_1, encode_2, labels, dist, max_val, min_val = cal_dist(model, tr_data_in_split, test_bs, metric,
                                                                  num_interval=num_interval, multi_gpu=multi_gpu)
    if verbose:
        logger.debug(f'max, min dist: {np.max(dist)}, {np.min(dist)}')
    threshold = np.linspace(np.min(dist)+1, np.max(dist)-1, 50)
    ret_stat = []
    for th in threshold[:]:
        stat = cal_stat(dist, th, labels)
        if stat[0] == 0 and stat[2] == 0:
            threshold = threshold[threshold != th]
            continue
        ret_stat.append(stat)
    try:
        prec_list = [tp / (tp + fp) for tp, tn, fp, fn in ret_stat]
        recall_list = [tp / (tp + fn) for tp, tn, fp, fn in ret_stat]
    except Exception:
        logger.info('error occurs during calculating pre/recall')
        return -1, 0.0, -1, -1
    # Add a small number to avoid dividing by 0
    prec_list = [x + 1e-6 if x == 0 else x for x in prec_list]
    recall_list = [x + 1e-6 if x == 0 else x for x in recall_list]
    f_score = np.array([2 / (1 / prec + 1 / recall) for prec, recall in zip(prec_list, recall_list)])

    ind = np.argmax(f_score)
    if verbose:
        logger.info(f'Training Acc: {(ret_stat[ind][0] + ret_stat[ind][1]) / (ret_stat[ind][0] + ret_stat[ind][1] + ret_stat[ind][2] + ret_stat[ind][3])}')
    return threshold[np.argmax(f_score)], np.max(f_score), max_val, min_val


def evaluate(model, test_data, th, max_val, min_val, test_bs, metric='hamming', num_interval=8, multi_gpu=False):
    encode_1, encode_2, labels, dist, _, _ = cal_dist(model, test_data, test_bs, metric, max_val, min_val,
                                                      num_interval=num_interval, multi_gpu=multi_gpu)
    tp, tn, fp, fn = cal_stat(dist, th, labels)
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    prec = prec + 1e-6 if prec == 0 else prec
    recall = recall + 1e-6 if recall == 0 else recall
    f_score = 2 / (1 / prec + 1 / recall)
    logger.info(f'Testing Acc: {acc}')
    logger.info(f'Testing threshold, f_score: {th}, {f_score}, prec: {prec}, recall: {recall}')
    logger.info(f'tp:{tp}, fn:{fn}, fp:{fp}, tn:{tn}')
    print(f'Testing Acc: {acc}')
    print(f'Testing threshold, f_score: {th}, {f_score}, prec: {prec}, recall: {recall}')
    print(f'tp:{tp}, fn:{fn}, fp:{fp}, tn:{tn}')
    return f_score


def cal_val_f_score(model, evaluate_data, test_bs, metric, num_interval, multi_gpu):
    th_dataset, val_dataset = evaluate_data
    threshold, f_score, max_val, min_val = cal_threshold(model, th_dataset, test_bs=test_bs,
                                                         metric=metric, num_interval=num_interval,
                                                         multi_gpu=multi_gpu, verbose=False)
    logger.info(f'Training f_score: {f_score}')
    val_f_score = evaluate(model, val_dataset, threshold, max_val, min_val, test_bs=test_bs,
                           metric=metric, num_interval=num_interval, multi_gpu=multi_gpu)
    return threshold, val_f_score, max_val, min_val


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def mean(self):
        try:
            return self.sum / self.count
        except Exception:
            return -1
