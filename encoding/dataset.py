import random
from itertools import product

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample
from torch.utils.data import Dataset

import log
from utils import df_to_str

logger = log.get_logger(__name__)

RAND_SEED = 42

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)


def preprocess(data1, data2, match, l_id_name, r_id_name):
    new_match = pd.DataFrame(columns=[l_id_name, r_id_name])
    for i in range(len(match)):
        l_id = data1.index[data1[l_id_name] == match.iloc[i][l_id_name]].tolist()[0]
        r_id = data2.index[data2[r_id_name] == match.iloc[i][r_id_name]].tolist()[0]
        new_match.loc[i] = [l_id, r_id]
    return new_match


def get_pos_neg_pairs(data1, data2, match):
    col_names = list(match.columns.values)
    data1 = data1.copy()
    data2 = data2.copy()
    data1['index_col'] = data1.index
    data2['index_col'] = data2.index
    interm_result = pd.merge(data1, match, how='inner', left_on='index_col', right_on=col_names[0])
    matching_table = pd.merge(interm_result, data2, how='inner', left_on=col_names[1], right_on='index_col')
    matching_table = matching_table[['index_col_x', 'index_col_y']]
    matching_pairs = list(matching_table.itertuples(index=False, name=None))
    # convert loc to ind
    for ind, (i, j) in enumerate(matching_pairs):
        matching_pairs[ind] = data1.index.get_loc(i), data2.index.get_loc(j)
    matching_pairs = set(matching_pairs)
    all_pairs = set(product(range(len(data1)), range(len(data2))))
    non_matching = all_pairs - matching_pairs
    return list(all_pairs), list(matching_pairs), list(non_matching)


def generate_testing_pairs(data1, data2, match, attribute, ratio=1, ret_labels=False, verbose=False):
    all_pairs, matching_pairs, non_matching = get_pos_neg_pairs(data1, data2, match)
    # data1 = data1.drop('index_col', axis=1)
    # data2 = data2.drop('index_col', axis=1)

    # converting tables to list of str
    data_l_str = df_to_str(data1, attribute)
    data_r_str = df_to_str(data2, attribute)

    positive_sample = []
    for i, j in matching_pairs:
        input1 = data_l_str[i]
        input2 = data_r_str[j]
        positive_sample.append(InputExample(texts=[input1, input2], label=1))
    if ratio == -1:
        non_matching_sample = non_matching
    else:
        random.seed(RAND_SEED)
        if len(positive_sample) * ratio > len(non_matching):
            logger.error('Error: # of positive pairs * ratio is larger than # of negative paris. Consider decreasing ratio.')
        non_matching_sample = random.sample(non_matching, len(positive_sample) * ratio)
    negative_sample = []
    for i, j in non_matching_sample:
        input1 = data_l_str[i]
        input2 = data_r_str[j]
        negative_sample.append(InputExample(texts=[input1, input2], label=0))
    if verbose:
        logger.info(f'num positive: {len(positive_sample)}')
        logger.info(f'num negative: {len(non_matching)}')
        logger.info(f'num used negative: {len(non_matching_sample)}')

    labels = None
    if ret_labels:
        labels = torch.zeros((len(data1), len(data2)), dtype=torch.int)
        for i in range(len(data1)):
            for j in range(len(data2)):
                if (i, j) in matching_pairs:
                    labels[i][j] = 1

    return positive_sample + negative_sample, labels, matching_pairs, non_matching_sample, data_l_str, data_r_str


def split(data1, data2, match, train_pos, val_pos, test_pos, attribute=None, tr_ratio=1, val_ratio=1, te_ratio=1, ret_labels=False, verbose=False):
    train_len_1 = int(len(data1) * train_pos)
    val_len_1 = int(len(data1) * val_pos)
    test_len_1 = int(len(data1) * test_pos)

    shuffle_data1 = data1.sample(frac=1, random_state=RAND_SEED)

    train_data_1 = shuffle_data1.iloc[:train_len_1]
    val_data_1 = shuffle_data1.iloc[train_len_1:(train_len_1+val_len_1)]
    test_data_1 = shuffle_data1.iloc[(train_len_1+val_len_1):]

    train_len_2 = int(len(data2) * train_pos)
    val_len_2 = int(len(data2) * val_pos)
    test_len_2 = int(len(data2) * test_pos)

    shuffle_data2 = data2.sample(frac=1, random_state=RAND_SEED)

    train_data_2 = shuffle_data2.iloc[:train_len_2]
    val_data_2 = shuffle_data2.iloc[train_len_2:(train_len_2 + val_len_2)]
    test_data_2 = shuffle_data2.iloc[(train_len_2 + val_len_2):]

    if verbose:
        logger.info('Generating dataset:')

    train_dataset, tr_labels, tr_match_ind, tr_non_match_ind, tr_data_l_str, tr_data_r_str = \
        generate_testing_pairs(train_data_1, train_data_2, match, attribute, tr_ratio, ret_labels, verbose=verbose)
    val_dataset, val_labels, val_match_ind, val_non_match_ind, val_data_l_str, val_data_r_str = \
        generate_testing_pairs(val_data_1, val_data_2, match, attribute, val_ratio, ret_labels=True, verbose=verbose)
    test_dataset, te_labels, te_match_ind, te_non_match_ind, te_data_l_str, te_data_r_str = \
        generate_testing_pairs(test_data_1, test_data_2, match, attribute, te_ratio, ret_labels=True, verbose=verbose)

    return (train_data_1, train_data_2, tr_labels, tr_match_ind, tr_non_match_ind, tr_data_l_str, tr_data_r_str), \
        (val_data_1, val_data_2, val_labels, val_match_ind, val_non_match_ind, val_data_l_str, val_data_r_str), \
        (test_data_1, test_data_2, te_labels, te_match_ind, te_non_match_ind, te_data_l_str, te_data_r_str), \
        train_dataset, val_dataset, test_dataset


# Dataset for Contras Learning such that a single batch contains exactly one pos.
class ContrastiveLearningPairwise(Dataset):
    def __init__(self, data1_str, data2_str, match_ind, non_match_ind, bs, world_size):
        self.data1 = data1_str
        self.data2 = data2_str
        self.match_ind = match_ind
        self.non_match_ind = non_match_ind
        self.bs = bs
        self.world_size = world_size

    def __getitem__(self, idx):
        if (idx // self.world_size) % (self.bs // self.world_size) == 0:
            ind1, ind2 = self.match_ind[idx//self.bs]  # return the same positive pair for different gpus
            return InputExample(texts=[self.data1[ind1], self.data2[ind2]], label=1)
        else:
            rand_ind = np.random.choice(len(self.non_match_ind))
            ind1, ind2 = self.non_match_ind[rand_ind]
            return InputExample(texts=[self.data1[ind1], self.data2[ind2]], label=0)

    def __len__(self):
        return len(self.match_ind) * self.bs

    def shuffle_pos(self):
        random.shuffle(self.match_ind)
