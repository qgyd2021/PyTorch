#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""
import argparse
from collections import defaultdict
from datetime import timedelta
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Callable, List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from datasets import load_dataset
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.bert.tokenization_bert import BertTokenizer

from project_settings import project_path
from toolbox.torch.modules.loss import FocalLoss
from toolbox.torch.training.metrics.categorical_accuracy import CategoricalAccuracy
from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torchtext.models.text_classification.text_cnn import TextCNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=(project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix(),
        type=str
    )

    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--max_cache_samples_count", default=1024, type=int)

    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--keep_most_recent_by_count", default=10, type=int)
    parser.add_argument("--patience", default=-1, type=int)
    parser.add_argument("--serialization_dir", default="serialization_dir", type=str)

    parser.add_argument("--lr_scheduler_step_size", default=50, type=int)
    parser.add_argument("--lr_scheduler_gamma", default=0.5, type=float)

    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--ddp_backend", default="gloo", type=str)
    parser.add_argument("--ddp_timeout", default=1800, type=int)

    args = parser.parse_args()
    return args


def distribute_setup(rank, world_size, ddp_backend: str = "nccl", ddp_timeout: int = 1800):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group(
        ddp_backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=ddp_timeout),
    )
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")


def distribute_cleanup():
    dist.destroy_process_group()


class TextGenerationJsonDataset(IterableDataset):
    def __init__(self,
                 text_key: str = "text",
                 ):
        self.text_key = text_key
        self.filename = None
        self.examples_stream = None

    def read(self, filename: str):
        self.filename = filename

    def __iter__(self):
        examples_stream = open(self.filename, "r", encoding="utf-8")
        self.examples_stream = examples_stream
        return self

    def __next__(self):
        row = next(self.examples_stream)
        row = json.loads(row)
        result = {
            "text": row[self.text_key]
        }
        return result


class GroupTextDataset(IterableDataset):
    def __init__(self, dataset: Dataset, tokenizer: BertTokenizer, max_seq_length: int, max_cache_samples_count: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_cache_samples_count = max_cache_samples_count

        self.cache_samples = list()
        self.cache_samples_count = 0

        self.current_iter_dataset = None
        self.current_sample = defaultdict(list)
        self.current_length = 0

    def __iter__(self):
        self.current_iter_dataset = iter(self.dataset)
        return self

    def __next__(self):
        if self.cache_samples_count > 0:
            sample = self.cache_samples.pop()
            self.cache_samples_count -= 1
            return sample
        else:
            while True:
                sample = next(self.current_iter_dataset)
                text = sample["text"]
                text_encoded = self.tokenizer(text)
                length = len(text_encoded[list(text_encoded.keys())[0]])
                self.current_length += length
                for k, v in text_encoded.items():
                    self.current_sample[k].extend(v)

                if self.current_length > self.max_seq_length:
                    sample = dict()
                    self.current_length -= self.max_seq_length
                    for k, v in self.current_sample.items():
                        sample[k] = v[:self.max_seq_length]
                        self.current_sample[k] = v[self.max_seq_length:]
                    self.cache_samples.append(sample)
                    self.cache_samples_count += 1
                if self.cache_samples_count >= self.max_cache_samples_count:
                    break


class CollateFunction(object):

    def __call__(self, batch: List[dict]):
        batch_ = defaultdict(list)
        for example in batch:
            for k, v in example.items():
                batch_[k].append(v[:-1])
            batch_["labels"] = example["input_ids"][1:]

        result = dict()
        for k, v in batch_.items():
            result[k] = torch.tensor(data=v, dtype=torch.long)
        return result


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))


def main():
    args = get_args()

    distribute_setup(
        args.rank,
        args.world_size,
        ddp_backend=args.ddp_backend,
        ddp_timeout=args.ddp_timeout,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("{}:{}".format(args.device, args.rank))
    n_gpu = torch.cuda.device_count()

    # model
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_name_or_path)
    model.to(device)

    # dataset
    train_dataset = TextGenerationJsonDataset()
    train_dataset.read(args.train_subset)
    train_dataset = GroupTextDataset(train_dataset, tokenizer, args.max_seq_length, args.max_cache_samples_count)
    valid_dataset = TextGenerationJsonDataset()
    valid_dataset.read(args.valid_subset)
    valid_dataset = GroupTextDataset(valid_dataset, tokenizer, args.max_seq_length, args.max_cache_samples_count)

    collate_fn = CollateFunction()

    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True
    )

    # train
    for batch in train_dataloader:
        print(batch)
        outputs = model.forward(**batch)
        print(outputs)
    # dist
    distribute_cleanup()
    return


if __name__ == '__main__':
    main()
