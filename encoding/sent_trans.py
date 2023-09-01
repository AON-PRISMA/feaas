import logging
import os
from collections import OrderedDict
from typing import Dict, Tuple, Iterable, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.losses import SiameseDistanceMetric
from sentence_transformers.util import batch_to_device
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import trange

import log
from utils import AverageMeter, cal_val_f_score

logger = log.get_logger(__name__)


def set_torch_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # set this for deterministic behavior for CUDA >= 10.2
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class ContrastiveLearningST(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        super(ContrastiveLearningST, self).__init__(*args, **kwargs)

    @staticmethod
    def get_dataloader(dataset, rank, world_size, batch_size, pin_memory=False, shuffle=True, num_workers=0):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                                drop_last=False, shuffle=False, sampler=sampler)

        return dataloader

    def fine_tune(self,
                  rank,
                  world_size,
                  ret_queue,
                  args,
                  train_data,
                  th_select_data,
                  val_data,
                  total_epochs,
                  optimizer_params: Dict[str, object] = {'lr': 2e-5},
                  train_dist='cos',
                  warmup_prop: int = 0,
                  temperature=1.0,
                  steps_per_epoch=None,
                  scheduler: str = 'WarmupLinear',
                  optimizer_class: Type[Optimizer] = torch.optim.AdamW,
                  weight_decay: float = 0.01,
                  max_grad_norm: float = 1,
                  show_progress_bar: bool = True,
                  ):
        self.to(self._target_device)
        contras_loss = {'cos': SiameseDistanceMetric.COSINE_DISTANCE,
                        'l1': SiameseDistanceMetric.MANHATTAN,
                        'l2': SiameseDistanceMetric.EUCLIDEAN}
        if args.multi_gpu:
            self._target_device = rank
        if args.mode == 'contras_pairwise':
            # no multi_gpu training for now
            train_loss = SupConLoss(model=self, distance_metric=contras_loss[train_dist],
                                    device=self._target_device,
                                    temperature=temperature)
        else:
            train_loss = losses.ContrastiveLoss(self, distance_metric=contras_loss[train_dist])
        if args.multi_gpu:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12345'
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

            # call log.reset_basic_config() to reset log.basicConfig (it seems dist.init_process_group overrides it)
            log_config_level = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
                                'error': logging.ERROR, 'critical': logging.CRITICAL}
            log_level = log_config_level[args.log_level]
            log.reset_basic_config(level=log_level, file_path=args.log_path)

            train_loader = ContrastiveLearningST.get_dataloader(train_data, rank, world_size, args.bs // world_size,
                                                                shuffle=True if args.mode == 'pairwise' else False)
            self._target_device = rank
            train_loss.to(rank)
            train_loss = DDP(train_loss, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        else:
            if args.mode in ['contras_pairwise', 'cp_margin_loss']:
                train_loader = DataLoader(train_data, batch_size=args.bs, sampler=torch.utils.data.SequentialSampler(train_data))
            else:
                train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = len(train_loader)
        num_train_steps = int(steps_per_epoch * total_epochs)
        warmup_steps = int(num_train_steps * warmup_prop)

        param_optimizer = list(train_loss.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                            t_total=num_train_steps)

        logger.info(f'num_train_steps: {num_train_steps}, warmup_steps: {warmup_steps}')
        best_f_score = 0
        best_e = 0
        best_model_dict = None
        for e in range(total_epochs):
            logger.info(f'Cur epoch: {e}')
            if args.multi_gpu:
                train_loader.sampler.set_epoch(e)
            # Tune the model for 1 epoch
            self.fit_custom(train_objectives=[(train_loader, train_loss)], optimizer_params={'lr': args.lr},
                            optimizer_obj=optimizer, scheduler_obj=scheduler_obj, epochs=1, cur_epoch=e,
                            max_grad_norm=max_grad_norm, show_progress_bar=show_progress_bar)
            if args.mode == 'contras_pairwise':
                train_data.shuffle_pos()  # shuffle the positive pairs in train_data each epoch
            if args.best_epoch and (rank == 0 or rank is None):
                _, val_f_score, _, _ = cal_val_f_score(self, (th_select_data, val_data), args.test_bs, args.metric,
                                                       args.num_interval, multi_gpu=False)  # use single gpu to evaluate
                if val_f_score > best_f_score:
                    best_f_score = val_f_score
                    best_e = e
                    best_model_dict = OrderedDict({k: v.to('cpu') for k, v in self.state_dict().items()})
            # logger.debug(f'lr: {scheduler_obj.get_last_lr()}')
        if args.best_epoch:
            ret_result = (best_e, best_f_score, best_model_dict)
        else:
            ret_result = (total_epochs-1, -1, OrderedDict({k: v.to('cpu') for k, v in self.state_dict().items()}))
        if args.multi_gpu and rank == 0:
            ret_queue.put(ret_result)
        if args.multi_gpu:
            dist.destroy_process_group()  # cleanup
        return ret_result

    def fit_custom(self,
                   train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
                   epochs: int = 1,
                   cur_epoch: int = None,
                   steps_per_epoch=None,
                   optimizer_obj=None,
                   scheduler_obj=None,
                   scheduler: str = 'WarmupLinear',
                   warmup_steps: int = 10000,
                   optimizer_class: Type[Optimizer] = torch.optim.AdamW,
                   optimizer_params: Dict[str, object] = {'lr': 2e-5},
                   weight_decay: float = 0.01,
                   max_grad_norm: float = 1,
                   use_amp: bool = False,
                   show_progress_bar: bool = True,
                   ):
        """
        Train the model with the given training objective
        Adapted from SentenceTransformer.fit()
        """

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        if optimizer_obj is None and scheduler_obj is None:
            num_train_steps = int(steps_per_epoch * epochs)

            # Prepare optimizers
            optimizers = []
            schedulers = []
            for loss_model in loss_models:
                param_optimizer = list(loss_model.named_parameters())

                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                     'weight_decay': weight_decay},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

                optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
                scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                    t_total=num_train_steps)

                optimizers.append(optimizer)
                schedulers.append(scheduler_obj)
        else:
            optimizers = [optimizer_obj]
            schedulers = [scheduler_obj]

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for e in range(epochs):
            training_steps = 0

            mean_loss = AverageMeter()

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()
            cur_epoch_ = e if cur_epoch is None else cur_epoch
            with trange(steps_per_epoch, desc=f"Epoch {cur_epoch_}:", smoothing=0.05, disable=not show_progress_bar) as pbar:
                for _ in pbar:
                    for train_idx in range(num_train_objectives):
                        loss_model = loss_models[train_idx]
                        optimizer = optimizers[train_idx]
                        scheduler = schedulers[train_idx]
                        data_iterator = data_iterators[train_idx]

                        try:
                            data = next(data_iterator)
                        except StopIteration:
                            data_iterator = iter(dataloaders[train_idx])
                            data_iterators[train_idx] = data_iterator
                            data = next(data_iterator)

                        features, labels = data
                        labels = labels.to(self._target_device)
                        features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))

                        if use_amp:
                            with autocast():
                                loss_value = loss_model(features, labels)

                            scale_before_step = scaler.get_scale()
                            scaler.scale(loss_value).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()

                            skip_scheduler = scaler.get_scale() != scale_before_step
                        else:
                            loss_value = loss_model(features, labels)
                            loss_value.backward()
                            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                            optimizer.step()

                        mean_loss.update(loss_value.detach().cpu().item())
                        pbar.set_description(f"Epoch {cur_epoch_}: Loss: {mean_loss.mean}")

                        optimizer.zero_grad()

                        if not skip_scheduler:
                            scheduler.step()

                training_steps += 1
                global_step += 1


class SupConLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, distance_metric, device, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.device = device
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2
        # the first pair is a positive pair, others are negative pairs
        assert labels[0] == 1
        assert torch.count_nonzero(labels[1:]).item() == 0
        rep_anchor, rep_other = reps
        # distances = self.distance_metric(rep_anchor, rep_other)
        # distances = F.cosine_similarity(rep_anchor, rep_other)
        similarity = torch.einsum('ij,ij->i', rep_anchor, rep_other)  # row-wise dot product
        similarity = torch.div(similarity, self.temperature)
        similarity = torch.unsqueeze(similarity, 0)  # add a dimension
        loss = F.cross_entropy(similarity, torch.tensor([0]).to(self.device), reduction='mean')
        return loss
