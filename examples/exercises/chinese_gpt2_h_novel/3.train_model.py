#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""
import argparse
from collections import defaultdict
import copy
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel

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
    parser.add_argument("--num_train_epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--keep_most_recent_by_count", default=3, type=int)
    parser.add_argument("--patience", default=-1, type=int)
    parser.add_argument("--serialization_dir", default="serialization_dir", type=str)

    # lr_scheduler StepLR
    parser.add_argument("--lr_scheduler_step_size", default=5000, type=int)
    parser.add_argument("--lr_scheduler_gamma", default=0.5, type=float)
    # lr_scheduler CosineAnnealingWarmRestarts
    parser.add_argument("--lr_scheduler_t_0", default=1, type=int)
    parser.add_argument("--lr_scheduler_t_mult", default=2, type=int)
    parser.add_argument("--lr_scheduler_eta_min", default=0., type=float)
    parser.add_argument("--lr_scheduler_last_epoch", default=-1, type=int)
    parser.add_argument("--lr_scheduler_verbose", action="store_true")

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


@dataclass
class TrainingArguments(object):
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    keep_most_recent_by_count: int = field(default=10, metadata={"help": "how many checkpoint to keep."})


@dataclass
class TrainerState(object):
    epoch_idx: int = 0
    global_step: int = 0

    best_epoch: int = -1
    best_validation_loss: float = float("inf")

    training_loss: float = float("inf")
    validation_loss: float = float("inf")


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class Trainer(object):
    def __init__(self,
                 model: PreTrainedModel,
                 args: TrainingArguments,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler,
                 device=None
                 ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # save checkpoint
        self.checkpoint_path_list = list()

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.device = device

        # global params
        self.state: TrainerState = None

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, dict):
            return dict({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, list):
            return list(self._prepare_input(v) for v in data)
        elif isinstance(data, tuple):
            return tuple(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        return data

    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = self._prepare_input(inputs)
        return inputs

    def train(self):
        self.model.to(self.device)

        self.state = TrainerState()

        for epoch_idx in range(self.args.num_train_epochs):
            self.state.epoch_idx = epoch_idx
            self.training_epoch(epoch_idx)
        return

    def training_epoch(self, epoch_idx: int):
        # train
        ddp_loss = torch.zeros(2).to(self.device)

        self.model.train()
        progress_bar = tqdm(self.train_dataloader, desc='Epoch={} (train)'.format(epoch_idx), leave=True)
        for step, batch in enumerate(progress_bar):
            batch = self._prepare_inputs(batch)
            loss, _ = self.training_step(batch)
            self.state.global_step += 1

            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

            # progress_bar
            training_loss: float = float(ddp_loss[0] / ddp_loss[1])
            progress_bar_postfix = {
                "loss": training_loss,
            }
            progress_bar.set_postfix(**progress_bar_postfix)

            # save
            dist.barrier()
            if self.state.global_step % self.args.save_steps == 0:
                self.save()

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        # progress_bar
        training_loss: float = float(ddp_loss[0] / ddp_loss[1])
        progress_bar_postfix = {
            "loss": training_loss,
        }
        progress_bar.set_postfix(**progress_bar_postfix)
        self.state.training_loss = training_loss

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      ) -> torch.Tensor:
        outputs = self.model.forward(**batch)
        loss = outputs.loss

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        return loss, outputs

    def save(self):
        this_checkpoint_path = self.output_dir / "checkpoint-{}".format(self.state.global_step)
        self.checkpoint_path_list.append(this_checkpoint_path)
        if len(self.checkpoint_path_list) > self.args.keep_most_recent_by_count:
            for checkpoint_path in self.checkpoint_path_list[:-self.args.keep_most_recent_by_count]:
                shutil.rmtree(checkpoint_path)
            self.checkpoint_path_list = self.checkpoint_path_list[-self.args.keep_most_recent_by_count:]
        self._save(this_checkpoint_path)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=False
        )
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        return


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
    if args.device == "cuda":
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

    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    # train
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_name_or_path)
    model.to(device)
    model = FSDP(model)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)
    # lr_scheduler = StepLR(
    #     optimizer=optimizer,
    #     step_size=args.lr_scheduler_step_size,
    #     gamma=args.lr_scheduler_gamma
    # )

    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=args.lr_scheduler_t_0,
        T_mult=args.lr_scheduler_t_mult,
        eta_min=args.lr_scheduler_eta_min,
        last_epoch=args.lr_scheduler_last_epoch,
        verbose=args.lr_scheduler_verbose,
    )

    # init_start_event.record()

    training_args = TrainingArguments(
        output_dir=args.serialization_dir,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        keep_most_recent_by_count=args.keep_most_recent_by_count,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )
    trainer.train()

    # init_end_event.record()

    # dist
    distribute_cleanup()
    return


if __name__ == '__main__':
    main()
