#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""
import argparse
from collections import defaultdict
import copy
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

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

    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--keep_most_recent_by_count", default=3, type=int)
    parser.add_argument("--patience", default=-1, type=int)
    parser.add_argument("--serialization_dir", default="serialization_dir", type=str)

    parser.add_argument("--lr_scheduler_step_size", default=50, type=int)
    parser.add_argument("--lr_scheduler_gamma", default=0.5, type=float)

    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--ddp_backend", default="nccl" if torch.cuda.is_available() else "gloo", type=str)
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
                    random.shuffle(self.cache_samples)
                    break


class CollateFunction(object):

    def __call__(self, batch: List[dict]):
        batch_ = defaultdict(list)
        for example in batch:
            if example is None:
                continue

            for k, v in example.items():
                batch_[k].append(v)
            batch_["labels"].append(example["input_ids"])

        result = dict()
        for k, v in batch_.items():
            result[k] = torch.tensor(data=v, dtype=torch.long)
        return result


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
    torch.cuda.set_device(device)
    n_gpu = torch.cuda.device_count()

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)

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

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # train
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_name_or_path)
    model.to(device)
    model = FSDP(model)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)
    lr_scheduler = StepLR(
        optimizer=optimizer,
        step_size=args.lr_scheduler_step_size,
        gamma=args.lr_scheduler_gamma
    )
    init_start_event.record()

    model_state_filename_list = list()
    training_state_filename_list = list()

    for epoch_idx in range(args.epochs):
        # train
        model.train()
        ddp_loss = torch.zeros(2).to(device)
        progress_bar = tqdm(train_dataloader, desc='Epoch={} (train)'.format(epoch_idx), leave=True)
        for step, batch in enumerate(progress_bar):
            for k, v in batch.items():
                batch[k] = v.to(device)

            outputs: CausalLMOutputWithCrossAttentions = model.forward(**batch)
            loss = outputs.loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

            # progress_bar
            progress_bar_postfix = {
                "loss": ddp_loss[0] / ddp_loss[1],
            }
            progress_bar.set_postfix(**progress_bar_postfix)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        training_loss: float = ddp_loss[0] / ddp_loss[1]
        if args.rank == 0:
            print("Train Epoch: {} \tLoss: {:.6f}".format(epoch_idx, ddp_loss[0] / ddp_loss[1]))

        # validation
        model.eval()
        ddp_loss = torch.zeros(2).to(device)
        progress_bar = tqdm(valid_dataloader, desc='Epoch={} (valid)'.format(epoch_idx), leave=True)
        with torch.no_grad():
            for step, batch in enumerate(progress_bar):
                for k, v in batch.items():
                    batch[k] = v.to(device)

                outputs: CausalLMOutputWithCrossAttentions = model.forward(**batch)
                loss = outputs.loss

                ddp_loss[0] += loss.item()
                ddp_loss[1] += 1

                # progress_bar
                progress_bar_postfix = {
                    "loss": ddp_loss[0] / ddp_loss[1],
                }
                progress_bar.set_postfix(**progress_bar_postfix)
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        validation_loss: float = ddp_loss[0] / ddp_loss[1]
        if args.rank == 0:
            print("Valid Epoch: {} \tLoss: {:.6f}".format(epoch_idx, ddp_loss[0] / ddp_loss[1]))

        # lr scheduler
        lr_scheduler.step()

        # serialization
        serialization_dir = Path(args.serialization_dir)
        serialization_dir.mkdir(exist_ok=True)

        # keep most recent by count
        if len(model_state_filename_list) >= args.keep_most_recent_by_count > 0:
            model_state_filename_ = model_state_filename_list.pop(0)
            os.remove(model_state_filename_)
            training_state_filename_ = training_state_filename_list.pop(0)
            os.remove(training_state_filename_)

        model_state_filename = serialization_dir / "model_state_{}.th".format(epoch_idx)
        training_state_filename = serialization_dir / "training_state_{}.th".format(epoch_idx)
        model_state_filename_list.append(model_state_filename.as_posix())
        with open(model_state_filename.as_posix(), "wb") as f:
            torch.save(model.state_dict(), f)
        training_state_filename_list.append(training_state_filename.as_posix())
        with open(training_state_filename.as_posix(), "wb") as f:
            torch.save(optimizer.state_dict(), f)

        # metrics epoch
        metrics = {
            "epoch": epoch_idx,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "lr": lr_scheduler.get_lr()
        }
        metrics_epoch_filename = serialization_dir / "metrics_epoch_{}.json".format(epoch_idx)
        with open(metrics_epoch_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

    init_end_event.record()

    # dist
    distribute_cleanup()
    return


if __name__ == '__main__':
    main()
