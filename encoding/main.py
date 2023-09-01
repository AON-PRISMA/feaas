import argparse
import json
import logging
import os

import torch
import torch.multiprocessing as mp

import log
from data_prep import data_prep
from dataset import split, ContrastiveLearningPairwise
from model import get_model
from sent_trans import set_torch_seed
from utils import evaluate, cal_val_f_score

logger = log.get_logger(__name__)


def run():
    model = get_model(args.model_name, encoding_size=args.encoding_size).to(device)
    tr_data_in_split, val_data_in_split, te_data_in_split, train_dataset, val_dataset, test_dataset = \
        split(data_l, data_r, data_match, 0.6, 0.2, 0.2, attribute=args.attr, tr_ratio=args.tr_ratio,
              val_ratio=args.val_ratio,
              te_ratio=args.te_ratio, ret_labels=False, verbose=True)
    train_data_l, train_data_r, tr_labels, tr_match_ind, tr_non_match_ind, tr_data_l_str, tr_data_r_str = tr_data_in_split
    val_data_l, val_data_r, val_labels, val_match_ind, val_non_match_ind, val_data_l_str, val_data_r_str = val_data_in_split
    test_data_l, test_data_r, te_labels, te_match_ind, te_non_match_ind, te_data_l_str, te_data_r_str = te_data_in_split

    # test_data_l.to_csv('../data/amazon_google/test_df_1.csv', index=False)
    # test_data_r.to_csv('../data/amazon_google/test_df_2.csv', index=False)
    # np.save('../data/amazon_google/te_labels.npy', te_labels)

    # check if Batch norm layers exist in the model
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if args.multi_gpu:
                logger.error('Batch norm layers found in the model, which affects multi-gpu training.')
                exit(1)
    world_size = torch.cuda.device_count()
    if not args.multi_gpu and world_size >= 1:
        world_size = 1
    logger.info(f'Using {world_size} gpu(s)')
    if args.multi_gpu:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # set TOKENIZERS_PARALLELISM to "False" to avoid deadlock

    th_select_data = tr_data_l_str, tr_data_r_str, tr_match_ind, tr_non_match_ind
    eval_data = val_data_l_str, val_data_r_str, val_match_ind, val_non_match_ind
    te_data = te_data_l_str, te_data_r_str, te_match_ind, te_non_match_ind
    if args.load_path is None:
        if args.mode == 'pairwise':
            train_data = train_dataset
        elif args.mode in ['contras_pairwise', 'cp_margin_loss']:
            train_data = ContrastiveLearningPairwise(tr_data_l_str, tr_data_r_str, tr_match_ind, tr_non_match_ind,
                                                     args.bs, world_size=world_size)
        else:
            raise NotImplementedError
        if args.multi_gpu:
            ret = torch.multiprocessing.Manager().Queue()
            mp.spawn(
                model.fine_tune,
                args=(world_size, ret, args, train_data, th_select_data, eval_data, args.epochs, {'lr': args.lr},
                      args.tr_dist, args.warmup_prop, args.temperature),
                nprocs=world_size,
                join=True
            )
            best_e, best_f_score, best_model_dict = ret.get()
        else:
            best_e, best_f_score, best_model_dict = model.fine_tune(None, None, None, args, train_data, th_select_data,
                                                                    eval_data, total_epochs=args.epochs,
                                                                    optimizer_params={'lr': args.lr},
                                                                    train_dist=args.tr_dist,
                                                                    warmup_prop=args.warmup_prop,
                                                                    temperature=args.temperature)

        logger.info(f'Best_f_score: {best_f_score}, Best_e: {best_e}')

        if args.log_path is not None:
            save_fn = os.path.join(log_dir, 'model.pth')
            torch.save(best_model_dict, save_fn)
        model.load_state_dict(best_model_dict)
    else:
        # search across the config files
        if 'run_' in args.load_path:
            load_fn = os.path.join(args.load_path, 'model.pth')
        else:
            load_fn = log.search_configs(args, args.load_path)
        if load_fn == '':
            print('Configs not found!')
            exit(1)
        print(f'Loading {load_fn}: ')
        model.load_state_dict(torch.load(load_fn))

    ratio_candidate = [1, 100, 500, 1000, 2000, -1]
    best_result = -1, -1, -1, -1
    for ratio in ratio_candidate:
        logger.info(f'Regenerating the training set ({ratio}) for selecting the threshold: ')
        try:
            tr_data_in_split, _, _, _, _, _ = \
                split(data_l, data_r, data_match, 0.6, 0.2, 0.2, attribute=args.attr, tr_ratio=ratio,
                      val_ratio=args.val_ratio,
                      te_ratio=args.te_ratio, ret_labels=False)
        except:
            logger.info(f'Skipping ratio {ratio}')  # The case that ratio is larger than max ratio of the dataset
            continue
        train_data_l, train_data_r, tr_labels, tr_match_ind, tr_non_match_ind, tr_data_l_str, tr_data_r_str = tr_data_in_split
        th_select_data = tr_data_l_str, tr_data_r_str, tr_match_ind, tr_non_match_ind
        result = cal_val_f_score(model, (th_select_data, eval_data), test_bs=args.test_bs, metric=args.metric,
                                 num_interval=args.num_interval, multi_gpu=args.multi_gpu)
        # logger.debug(f'max_val, min_val (based on the training data): {max_val}, {min_val}')
        # logger.info(f'Training threshold, f_score: {threshold}, {f_score}')
        logger.info(f'val_f_score: {result[1]}, threshold: {result[0]}')
        if result[1] > best_result[1]:
            best_result = result
    logger.info(f'best_f_score: {best_result[1]}, best_threshold: {best_result[0]}')

    evaluate(model, te_data, best_result[0], max_val=best_result[2], min_val=best_result[3], test_bs=args.test_bs,
             metric=args.metric, num_interval=args.num_interval, multi_gpu=args.multi_gpu)


def args_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--load_config', type=str, default=None, help='the config file')

    # command line arguments
    # Training arguments:
    group.add_argument('-d', '--dataset_name', type=str, default='ag', choices=['ag', 'febrl4', 'abt_buy', 'febrl_gen'],
                       help="the dataset set name used in training")
    parser.add_argument('-n', '--model_name', type=str, default='all-distilroberta-v1',
                        help="the model architecture name")
    parser.add_argument('-s', '--encoding_size', type=int, default=256,
                        help="the dimension of the output encoding vector")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('--bs', type=int, default=32, help="training batch size")
    parser.add_argument('--warmup_prop', type=float, default=0.0, help="proportion of steps for lr warmup")
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature used in info_nce and sup_con loss")
    parser.add_argument('--tr_dist', type=str, default='cos', help="distance metric for similarity during training")
    parser.add_argument('--attr', type=str, default=None, help="attributes used from the dataset; None for use all")
    parser.add_argument('--mode', type=str, default='pairwise', choices=['pairwise', 'contras_pairwise', 'cp_margin_loss'],
                        help="name of method to be used in training")
    parser.add_argument('--tr_ratio', type=int, default=1, help="the imbalance ratio of the training data")

    # Other arguments:
    parser.add_argument('--val_ratio', type=int, default=1, help="the imbalance ratio of the validation data")
    parser.add_argument('--te_ratio', type=int, default=1, help="the imbalance ratio of the test data")

    parser.add_argument('--test_bs', type=int, default=512, help="testing batch size")

    parser.add_argument('--num_interval', type=int, default=8, help="number of intervals for quantization")
    parser.add_argument('--metric', type=str, default='hamming', help="distance metric used in testing")
    parser.add_argument('--best_epoch', type=int, default=1, help="select the best epoch using the validation data")
    parser.add_argument('--multi_gpu', type=int, default=0, help="use multi gpus or not")

    parser.add_argument('--load_path', type=str, default=None,
                        help="path for loading the config file and model checkpoint")

    parser.add_argument('--log_level', type=str, default='debug', help="logging level")
    parser.add_argument('--log_path', type=str, default='saved_results',
                        help="logging file path; None for not logging to file")

    parser.add_argument('--seed', type=int, default=None, help="random seed for pytorch training")

    args = parser.parse_args()
    # load json file
    if args.load_config is not None:
        with open(args.load_config, 'rt') as fh:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(fh))
            args = parser.parse_args(namespace=t_args)

    return args


if __name__ == "__main__":
    args = args_parser()
    if args.log_path == 'None':
        args.log_path = None
    if args.seed == 'None':
        args.seed = None
    log_config_level = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
                        'error': logging.ERROR, 'critical': logging.CRITICAL}
    log_level = log_config_level[args.log_level]

    if args.load_path is not None:  # do not log to file when loading a model
        log_dir = log.init(log_level, None)
    else:
        log_dir = log.init(log_level, args.log_path)
    print(f'Logging dir: {log_dir}')
    # Ignore this library's INFO level logging
    logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.WARNING)

    # Logging the config
    args_dict = vars(args)
    print(args_dict)
    args_str = json.dumps(args_dict, indent=4)
    # logger.info(args_str)
    if args.load_path is None and args.log_path is not None:
        with open(os.path.join(log_dir, "config.json"), 'w') as f:
            json.dump(args_dict, f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_l, data_r, data_match = data_prep(args.dataset_name)

    if args.seed is not None:
        set_torch_seed(args.seed)

    run()
    logger.info('end')
