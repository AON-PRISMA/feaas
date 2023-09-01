import logging
import os
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger('main')


def get_data(args):
    cur_path = Path(__file__).parent.resolve()
    data_path = os.path.join(cur_path, args.data_path)
    data = np.load(data_path)[:]
    logger.debug(f'Loaded data shape: {data.shape}')
    datalist: list = data.tolist()
    record = {}
    for i in range(len(datalist)):
        record[i] = {'data_record': datalist[i], 'key_id': 0}
    return record


def get_test_data(args):
    with open(os.path.join(Path(__file__).parent.resolve(), f'tests/test_{args.test_data}/records.pkl'), 'rb') as f:
        records = pickle.load(f)

    return records[args.cid]


def gen_test_data():
    data_1 = [i // 32 for i in range(256)]
    data_2 = [i // 32 for i in reversed(range(256))]
    data_3 = [i // 32 + 1 if i // 32 + 1 < 8 else 0 for i in range(256)]

    # test 1:
    num_clients = 3
    record_1 = {0: {'data_record': data_1, 'key_id': 0},
                1: {'data_record': data_2, 'key_id': 0},
                2: {'data_record': data_3, 'key_id': 0, "last_record": True}}

    record_2 = {0: {'data_record': data_2, 'key_id': 0},
                1: {'data_record': data_1, 'key_id': 0},
                2: {'data_record': data_3, 'key_id': 0, "last_record": True}}

    record_3 = {0: {'data_record': data_3, 'key_id': 0},
                1: {'data_record': data_3, 'key_id': 0},
                2: {'data_record': data_3, 'key_id': 0, "last_record": True}}

    records = {1: record_1, 2: record_2, 3: record_3}
    ground_truth = np.zeros((int(num_clients * (num_clients-1) / 2 * 9), 5))

    counter = 0
    for c1 in range(2, num_clients+1):
        for c2 in range(1, c1):
            for i in range(len(records[c1])):
                for j in range(len(records[c2])):
                    if (c1, c2, i, j) in [(2, 1, 1, 0), (2, 1, 2, 2), (2, 1, 0, 1), (3, 2, 0, 2),
                                          (3, 2, 1, 2), (3, 2, 2, 2), (3, 1, 0, 2), (3, 1, 1, 2),
                                          (3, 1, 2, 2)]:
                        pred = True
                    else:
                        pred = False
                    ground_truth[counter] = [c1, i+1, c2, j+1, pred]
                    counter += 1

    Path("tests/test_1/").mkdir(parents=True, exist_ok=True)
    with open('tests/test_1/records.pkl', 'wb') as f:
        pickle.dump(records, f)

    np.save('tests/test_1/ground_truth.npy', ground_truth)

    # test 2:
    num_clients = 3
    record_1 = {0: {'data_record': data_1, 'key_id': 0},
                1: {'data_record': data_2, 'key_id': 0},
                2: {'data_record': data_1, 'key_id': 0, "last_record": True}}

    record_2 = {0: {'data_record': data_2, 'key_id': 0},
                1: {'data_record': data_1, 'key_id': 0},
                2: {'data_record': data_3, 'key_id': 0, "last_record": True}}

    record_3 = {0: {'data_record': data_3, 'key_id': 0},
                1: {'data_record': data_3, 'key_id': 0},
                2: {'data_record': data_3, 'key_id': 0, "last_record": True}}

    records = {1: record_1, 2: record_2, 3: record_3}
    ground_truth = np.zeros((int(num_clients * (num_clients-1) / 2 * 9), 5))

    counter = 0
    for c1 in range(2, num_clients+1):
        for c2 in range(1, c1):
            for i in range(len(records[c1])):
                for j in range(len(records[c2])):
                    if (c1, c2, i, j) in [(2, 1, 1, 0), (2, 1, 1, 2), (2, 1, 0, 1), (3, 2, 0, 2),
                                          (3, 2, 1, 2), (3, 2, 2, 2)]:
                        pred = True
                    else:
                        pred = False
                    ground_truth[counter] = [c1, i+1, c2, j+1, pred]
                    counter += 1

    Path("tests/test_2/").mkdir(parents=True, exist_ok=True)
    with open('tests/test_2/records.pkl', 'wb') as f:
        pickle.dump(records, f)

    np.save('tests/test_2/ground_truth.npy', ground_truth)


if __name__ == "__main__":
    gen_test_data()
