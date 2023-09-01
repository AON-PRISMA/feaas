import argparse
import csv
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_name', type=str, default='ag')
    parser.add_argument('--test_data', type=int, default=0)

    parser.add_argument('--th', type=int, help='the decision making threshold')
    parser.add_argument('--rep', type=int, default=1, help='repetition param for the encryption')
    parser.add_argument('--lsh_rep', type=int, default=10, help='repetition param for lsh')
    parser.add_argument('--lsh_num_ind', type=int, default=7, help='number of selected indices for lsh')
    parser.add_argument('--lsh_num_bin', type=int, default=16, help='number of bins for lsh')

    parser.add_argument('--plain', type=int, default=0, help='send plain texts directly or not. For testing purposes')

    parser.add_argument('--num_bits', type=int, default=32, help='number of bits')

    parser.add_argument('--tls', type=int, default=0, help='whether to enable tls')
    parser.add_argument('--debug', type=int, default=0, help='whether to enable debug mode')

    return parser.parse_args()


args = args_parser()
print(args)

if args.test_data != 0:
    with open(os.path.join(Path(__file__).parent.resolve(), f'../client/tests/test_{args.test_data}/records.pkl'), 'rb') as f:
        records = pickle.load(f)

    num_clients = len(records)

    ground_truth = np.load(f'../client/tests/test_{args.test_data}/ground_truth.npy')
    embedding_path = ''
else:
    num_clients = 2

    cmd_to_data_dict = {'ag': 'amazon_google', 'febrl4': 'febrl4', 'abt_buy': 'abt_buy'}
    embedding_path = f'../data/{cmd_to_data_dict[args.data_name]}/'
    ground_truth = np.load(embedding_path + 'te_labels.npy')

    os.chdir('../encoding/')
    for cid in range(num_clients):
        proc_c = subprocess.Popen(["python", "encode.py", "--load_config", embedding_path + 'encoding_config.json',
                                   '-d', embedding_path + f'test_df_{cid+1}.csv', '-s', f'../client/embed_{cid+1}.npy'])
        proc_c.communicate()

    os.chdir('../server/')


print(f'num_clients: {num_clients}')

proc1 = subprocess.Popen(["./server", f'-num_c={num_clients}', f'-rep={args.rep}',
                          f'-lsh_rep={args.lsh_rep}', f'-plain={args.plain}', f'-tls={args.tls}', f'-debug={args.debug}'])
print('proc_s = ', proc1.pid)
time.sleep(1)

proc_list = []

for cid in range(num_clients):
    while True:
        if args.test_data != 0:
            proc_c = subprocess.Popen(["python", "../client/main.py", '--cid', str(cid+1), '--test_data', f'{args.test_data}',
                                       '--th', f'{args.th}', '--rep', f'{args.rep}', '--lsh_rep', f'{args.lsh_rep}',
                                       '--lsh_num_ind', f'{args.lsh_num_ind}', '--lsh_num_bin', f'{args.lsh_num_bin}', '--num_bits', f'{args.num_bits}',
                                       '--plain', f'{args.plain}', '--tls_file', 'rootCA.pem' if args.tls else '', '--debug', f'{args.debug}'])
        else:
            proc_c = subprocess.Popen(["python", "../client/main.py", '--cid', str(cid+1), '--load_config', embedding_path + 'config.json',
                                       '--data_path', f'../client/embed_{cid+1}.npy', '--test_data', f'{args.test_data}',
                                       '--th', f'{args.th}', '--rep', f'{args.rep}', '--lsh_rep', f'{args.lsh_rep}',
                                       '--lsh_num_ind', f'{args.lsh_num_ind}', '--lsh_num_bin', f'{args.lsh_num_bin}', '--num_bits', f'{args.num_bits}',
                                       '--plain', f'{args.plain}', '--tls_file', 'rootCA.pem' if args.tls else '', '--debug', f'{args.debug}'])
        time.sleep(1)
        ret_code = proc_c.poll()
        if ret_code is None:
            break
        else:
            print('client fails to connect. Waiting for the server to start...')
            time.sleep(20)

    print(f'proc_c{cid+1} = {proc_c.pid}')
    proc_list.append(proc_c)
    time.sleep(0.5)


for p in proc_list:
    p.communicate()

proc1.kill()
for p in proc_list:
    p.kill()


# calculate the accuracy
tp, fn, fp, tn = 0, 0, 0, 0

with open('result.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, )
    next(reader, None)  # ignore the header
    counter = 0
    for row in reader:
        if args.test_data:
            assert all(int(x) == int(y) for x, y in zip(row[:-1], ground_truth[counter][:-1]))
            if row[4] == '1' and ground_truth[counter][-1]:
                tp += 1
            elif row[4] == '1' and not ground_truth[counter][-1]:
                fp += 1
            elif row[4] == '0' and not ground_truth[counter][-1]:
                tn += 1
            elif row[4] == '0' and ground_truth[counter][-1]:
                fn += 1
        else:
            ind1, ind2 = int(row[3]) - 1, int(row[1]) - 1
            if row[4] == '1' and ground_truth[ind1][ind2]:
                tp += 1
            elif row[4] == '1' and not ground_truth[ind1][ind2]:
                fp += 1
            elif row[4] == '0' and not ground_truth[ind1][ind2]:
                tn += 1
            elif row[4] == '0' and ground_truth[ind1][ind2]:
                fn += 1
        counter += 1
    print(f'tp:{tp}, fn:{fn}, fp:{fp}, tn:{tn}')
    print(2*tp / (2*tp+fp+fn))
    print((tp+tn) / (tp+tn+fp+fn))
