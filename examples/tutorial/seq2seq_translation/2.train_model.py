#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import math
import numpy as np
import os
from pathlib import Path
import platform
import random
import time

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from toolbox.open_nmt.data.collate_functions.collate_fn import CommonCollateFunction
from toolbox.open_nmt.data.dataset.dataset import PairDataset
from toolbox.open_nmt.data.preprocess.preprocess import LowercasePreprocess, ReversePreprocess, UnicodeToAsciiPreprocess
from toolbox.open_nmt.data.sample_filter.sample_filter import LengthFilter, PrefixFilter
from toolbox.open_nmt.data.dataset.preprocess_filter_dataset import PreprocessFilterDataset
from toolbox.open_nmt.data.tokenizers.tokenizer import WhitespaceTokenizer
from toolbox.open_nmt.data import vocabulary
from toolbox.open_nmt.modules.encoders.gru_encoder import GRUEncoder
from toolbox.open_nmt.modules.decoders.gru_decoder import GruAttentionDecoder
from toolbox.open_nmt.models.seq2seq_translation import Seq2SeqTranslation
from toolbox.open_nmt.training.metrics.bleu import BLEU


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_file", default="data_dir/data/eng-fra.txt", type=str)

    parser.add_argument("--vocabulary_dir", default="./vocabulary", type=str)

    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_train_epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--keep_most_recent_by_count", default=10, type=int)
    parser.add_argument("--patience", default=-1, type=int)
    parser.add_argument("--serialization_dir", default="serialization_dir", type=str)

    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--device_id", default=0, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("{}:{}".format(args.device, args.device_id))
    n_gpu = torch.cuda.device_count()

    vocabulary.DEFAULT_PADDING_TOKEN = "<pad>"
    vocabulary.DEFAULT_OOV_TOKEN = "<unk>"
    vocab = vocabulary.Vocabulary.from_files(args.vocabulary_dir)

    pad_idx = 0
    bos_idx = 1
    eos_idx = 2
    unk_idx = 3
    special_symbols = ["<pad>", "<bos>", "<eos>", "<unk>"]

    model = Seq2SeqTranslation(
        encoder=GRUEncoder(
            vocab_size=vocab.get_vocab_size(namespace="src_tokens"),
            hidden_size=128,
        ),
        decoder=GruAttentionDecoder(
            max_decoding_length=15,
            hidden_size=128,
            vocab_size=vocab.get_vocab_size(namespace="tgt_tokens"),
            bos_idx=bos_idx
        ),
    )
    model.to(device)

    dataset = PairDataset()
    dataset.read(args.pair_file)
    dataset = PreprocessFilterDataset(
        dataset=dataset,
        preprocess_list=[
            LowercasePreprocess(),
            ReversePreprocess(),
            UnicodeToAsciiPreprocess()
        ],
        sample_filter_list=[
            LengthFilter(max_length=10),
            PrefixFilter(tgt_prefixes=[
                "i am ", "i m ",
                "he is", "he s ",
                "she is", "she s ",
                "you are", "you re ",
                "we are", "we re ",
                "they are", "they re ",
            ])
        ]
    )

    collate_fn = CommonCollateFunction(
        src_tokenizer=WhitespaceTokenizer(),
        tgt_tokenizer=WhitespaceTokenizer(),
        vocabulary=vocab,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if platform.system() == "Windows" else os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.NLLLoss()
    metric = BLEU(exclude_indices=[0, 1, 2])

    # serialization
    serialization_dir = Path(args.serialization_dir)
    serialization_dir.mkdir(exist_ok=True)

    # global params
    best_epoch: int = -1
    best_validation_accuracy: float = -1
    best_validation_loss: float = float("inf")

    model_state_filename_list = list()
    training_state_filename_list = list()

    for epoch_idx in range(1, args.num_train_epochs + 1):
        metric.reset()
        training_total_loss: float = 0.0
        training_total_steps: int = 0

        progress_bar = tqdm(train_dataloader, desc="Epoch={} (train)".format(epoch_idx), leave=True)
        for step, batch in enumerate(progress_bar):
            inputs, targets = batch

            optimizer.zero_grad()

            outputs = model.forward(inputs, targets)
            logits = outputs["logits"]

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            loss.backward()

            optimizer.step()

            predictions = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(predictions, dim=-1)
            metric.__call__(predictions, gold_targets=targets)

            training_total_loss += loss.item()
            training_total_steps += 1

            # progress_bar
            progress_bar_postfix = {
                "loss": training_total_loss / training_total_steps,
                "bleu": round(metric.get_metric()["BLEU"], 4)
            }
            progress_bar.set_postfix(**progress_bar_postfix)

        training_loss: float = training_total_loss / training_total_steps

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

    return


if __name__ == '__main__':
    main()
