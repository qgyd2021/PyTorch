#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import platform
import random
from typing import List

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import load_dataset
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizer

from project_settings import project_path
from toolbox.torch.modules.loss import FocalLoss
from toolbox.torch.training.metrics.categorical_accuracy import CategoricalAccuracy
from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="bitext/customer-support-intent-dataset", type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )

    parser.add_argument(
        "--pretrained_model_dir",
        default=(project_path / "pretrained_models/bert-base-uncased").as_posix(),
        type=str
    )

    parser.add_argument("--vocabulary_dir", default="./vocabulary", type=str)

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
    parser.add_argument("--device_id", default=0, type=int)

    args = parser.parse_args()
    return args


class CollateFunction(object):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 text_field: str = "text",
                 label_field: str = "label"
                 ):
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.label_field = label_field

    def __call__(self, batch: List[dict]):
        texts: List[str] = list()
        targets: List[int] = list()
        for example in batch:
            text = example[self.text_field]
            label = example[self.label_field]
            texts.append(text)
            targets.append(label)

        encodings = self.tokenizer.__call__(
            text=texts,
            padding="longest",
            truncation=True,
            max_length=512,
        )
        inputs = encodings["input_ids"]

        inputs = np.array(inputs)
        targets = np.array(targets)

        inputs: torch.LongTensor = torch.from_numpy(inputs)
        targets: torch.LongTensor = torch.from_numpy(targets)
        return inputs, targets


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("{}:{}".format(args.device, args.device_id))
    n_gpu = torch.cuda.device_count()

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    categorical_accuracy = CategoricalAccuracy(top_k=1)
    categorical_accuracy5 = CategoricalAccuracy(top_k=5)

    loss_fn = FocalLoss(vocabulary.get_vocab_size(namespace="labels"))

    dataset = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        cache_dir=args.dataset_cache_dir
    )
    print(dataset)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)

    collate_fn = CollateFunction(
        tokenizer=tokenizer,
        text_field="utterance",
        label_field="intent"
    )

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if platform.system() == "Windows" else os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )
    valid_dataloader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if platform.system() == "Windows" else os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )

    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_dir
    )
    model.to(device)

    # serialization
    serialization_dir = Path(args.serialization_dir)
    serialization_dir.mkdir(exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    lr_scheduler = StepLR(
        optimizer=optimizer,
        step_size=args.lr_scheduler_step_size,
        gamma=args.lr_scheduler_gamma
    )

    # global params
    best_epoch: int = -1
    best_validation_accuracy: float = -1
    best_validation_loss: float = float("inf")

    model_state_filename_list = list()
    training_state_filename_list = list()

    for epoch_idx in range(args.epochs):

        # train
        categorical_accuracy.reset()
        training_total_loss: float = 0.0
        training_total_steps: int = 0

        model.train()
        progress_bar = tqdm(train_dataloader, desc="Epoch={} (train)".format(epoch_idx), leave=True)
        for step, batch in enumerate(progress_bar):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets: torch.LongTensor = targets.to(device).long()

            logits = model.forward(inputs)

            loss: torch.Tensor = loss_fn.forward(logits, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            predictions = torch.softmax(logits, dim=-1)
            categorical_accuracy(predictions, gold_labels=targets)
            categorical_accuracy5(predictions, gold_labels=targets)

            training_total_loss += loss.item()
            training_total_steps += 1

            # progress_bar
            progress_bar_postfix = {
                "loss": training_total_loss / training_total_steps,
                "accuracy": categorical_accuracy.get_metric(),
                "accuracy5": categorical_accuracy5.get_metric(),
            }
            progress_bar.set_postfix(**progress_bar_postfix)
        training_accuracy: float = categorical_accuracy.get_metric()
        training_accuracy5: float = categorical_accuracy5.get_metric()
        training_loss: float = training_total_loss / training_total_steps

        # validation
        categorical_accuracy.reset()
        validation_total_loss: float = 0.0
        validation_total_steps: int = 0

        model.eval()
        progress_bar = tqdm(valid_dataloader, desc="Epoch={} (valid)".format(epoch_idx), leave=True)
        with torch.no_grad():
            for step, batch in enumerate(progress_bar):
                inputs, targets = batch
                inputs = inputs.to(device)
                targets: torch.LongTensor = targets.to(device).long()

                logits = model.forward(inputs)

                loss: torch.Tensor = loss_fn.forward(logits, targets)

                predictions = torch.softmax(logits, dim=-1)
                categorical_accuracy(predictions, gold_labels=targets)

                validation_total_loss += loss.item()
                validation_total_steps += 1

                # progress_bar
                progress_bar_postfix = {
                    "loss": validation_total_loss / validation_total_steps,
                    "accuracy": categorical_accuracy.get_metric(),
                    "accuracy5": categorical_accuracy5.get_metric(),
                }
                progress_bar.set_postfix(**progress_bar_postfix)
        validation_accuracy = categorical_accuracy.get_metric()
        validation_accuracy5 = categorical_accuracy5.get_metric()
        validation_loss: float = validation_total_loss / validation_total_steps

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

        # early stop
        if validation_accuracy > best_validation_accuracy:
            with open(serialization_dir / "best.th", "wb") as f:
                torch.save(model.state_dict(), f)

            best_epoch = epoch_idx
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss

        if epoch_idx - best_epoch >= args.patience > 0:
            # early stop
            break

        # metrics epoch
        metrics = {
            "best_epoch": best_epoch,

            "epoch": epoch_idx,

            "training_accuracy": training_accuracy,
            "training_accuracy5": training_accuracy5,
            "training_loss": training_loss,

            "validation_accuracy": validation_accuracy,
            "validation_accuracy5": validation_accuracy5,
            "validation_loss": validation_loss,

            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_loss": best_validation_loss,
        }
        metrics_epoch_filename = serialization_dir / "metrics_epoch_{}.json".format(epoch_idx)
        with open(metrics_epoch_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

    return


if __name__ == '__main__':
    main()
