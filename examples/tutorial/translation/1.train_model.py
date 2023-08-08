#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""
import argparse
import math
import os
import pickle
import platform
import random
from typing import Any, Callable, Dict, Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi30k_train_url",
        default="https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
        type=str
    )
    parser.add_argument(
        "--multi30k_valid_url",
        default="https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
        type=str
    )
    parser.add_argument("--src_language", default="de", type=str)
    parser.add_argument("--tgt_language", default="en", type=str)

    parser.add_argument("--src_tokenizer_name", default="spacy", type=str)
    parser.add_argument("--src_tokenizer_language", default="de_core_news_sm", type=str)
    parser.add_argument("--tgt_tokenizer_name", default="spacy", type=str)
    parser.add_argument("--tgt_tokenizer_language", default="en_core_web_sm", type=str)

    parser.add_argument("--src_vocab_pkl", default="vocab_de.pkl", type=str)
    parser.add_argument("--tgt_vocab_pkl", default="vocab_en.pkl", type=str)

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

    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--ddp_backend", default="gloo", type=str)
    parser.add_argument("--ddp_timeout", default=1800, type=int)

    args = parser.parse_args()
    return args


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 num_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=num_head,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        """
        :param src: shape=[src_seq_len, batch_size].
        :param trg: shape=[trg_seq_len, batch_size].
        :param src_mask:
        :param tgt_mask:
        :param src_padding_mask:
        :param tgt_padding_mask:
        :param memory_key_padding_mask:
        :return: logits, shape=[batch_size, tgt_vocab_size, tgt_out_seq_len]
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        logits: torch.Tensor = self.generator(outs)
        return logits

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        memory = self.transformer.encoder(
            self.positional_encoding(
                self.src_tok_emb(src)
            ),
            src_mask
        )
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt_out = self.transformer.decoder(
            self.positional_encoding(
                self.tgt_tok_emb(tgt)
            ),
            memory,
            tgt_mask
        )
        return tgt_out


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(size=(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: torch.Tensor,
                tgt: torch.Tensor,
                pad_idx: int,
                ):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class CollateFunction(object):
    def __init__(self,
                 src_tokenizer,
                 tgt_tokenizer,
                 src_vocab: Vocab,
                 tgt_vocab: Vocab,
                 bos_idx: int,
                 pad_idx: int,
                 ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx

    @staticmethod
    def make_mask(
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        pad_idx: int,
    ):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt_input.shape[0]

        mask = torch.triu(torch.ones(size=(tgt_seq_len, tgt_seq_len)) == 1).transpose(0, 1)
        tgt_mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

        src_padding_mask = (src == pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt_input == pad_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def __call__(self, batch: List[tuple]):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_tokens = self.src_tokenizer(src_sample)
            src_token_ids = self.src_vocab(src_tokens)
            src_input_ids = torch.cat(
                tensors=(
                    torch.tensor([self.bos_idx]),
                    torch.tensor(src_token_ids),
                    torch.tensor([self.pad_idx])
                )
            )

            tgt_tokens = self.tgt_tokenizer(tgt_sample)
            tgt_token_ids = self.tgt_vocab(tgt_tokens)
            tgt_input_ids = torch.cat(
                tensors=(
                    torch.tensor([self.bos_idx]),
                    torch.tensor(tgt_token_ids),
                    torch.tensor([self.pad_idx])
                )
            )

            src_batch.append(src_input_ids)
            tgt_batch.append(tgt_input_ids)

        # [seq_len, batch_size]
        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_idx)

        tgt_input = tgt_batch[:-1, :]
        tgt_out = tgt_batch[1:, :]

        # mask
        # src_mask.shape = [src_seq_len, batch_size]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.make_mask(
            src_batch, tgt_input, pad_idx=self.pad_idx
        )

        inputs = {
            "src": src_batch,
            "trg": tgt_input,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "src_padding_mask": src_padding_mask,
            "tgt_padding_mask": tgt_padding_mask,
            "memory_key_padding_mask": src_padding_mask,
        }
        return inputs, tgt_out


class Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 # compute_metrics: Callable,
                 loss_fn: Callable,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 max_epoch: int = 200,
                 device=None
                 ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.max_epoch = max_epoch
        self.device = device

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

        for epoch_idx in range(self.max_epoch):
            self.training_epoch(epoch_idx)
            self.evaluation_epoch(epoch_idx)

        return

    def training_epoch(self, epoch_idx: int):
        # train
        training_total_loss: float = 0.0
        training_total_steps: int = 0

        progress_bar = tqdm(self.train_dataloader, desc='Epoch={} (train)'.format(epoch_idx), leave=True)
        for step, batch in enumerate(progress_bar):
            self.model.train()
            inputs, targets = batch
            loss, _ = self.training_step(inputs, targets)

            training_total_loss += loss.item()
            training_total_steps += 1

            # progress_bar
            progress_bar_postfix = {
                "loss": training_total_loss / training_total_steps,
            }
            progress_bar.set_postfix(**progress_bar_postfix)

        return

    def training_step(self,
                      inputs: Dict[str, torch.Tensor],
                      targets: torch.Tensor
                      ) -> torch.Tensor:
        inputs = self._prepare_inputs(inputs)

        logits = self.model.forward(**inputs)

        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        return loss, logits

    def evaluation_epoch(self, epoch_idx: int):
        # validation
        validation_total_loss: float = 0.0
        validation_total_steps: int = 0

        progress_bar = tqdm(self.valid_dataloader, desc='Epoch={} (valid)'.format(epoch_idx), leave=True)
        for step, batch in enumerate(progress_bar):
            self.model.train()
            inputs, targets = batch
            loss, _ = self.prediction_step(inputs, targets)

            validation_total_loss += loss.item()
            validation_total_steps += 1

            # progress_bar
            progress_bar_postfix = {
                "loss": validation_total_loss / validation_total_steps,
            }
            progress_bar.set_postfix(**progress_bar_postfix)

        return

    def prediction_step(self,
                        inputs: Dict[str, torch.Tensor],
                        targets: torch.Tensor
                        ) -> torch.Tensor:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            logits = self.model.forward(**inputs)

            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        return loss, logits


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("{}:{}".format(args.device, args.device_id))
    n_gpu = torch.cuda.device_count()

    unk_idx = 0
    pad_idx = 1
    bos_idx = 2
    eos_idx = 3
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

    multi30k.URL["train"] = args.multi30k_train_url
    multi30k.URL["valid"] = args.multi30k_valid_url

    with open(args.src_vocab_pkl, "rb") as f:
        src_vocab = pickle.load(f)
    src_vocab = Vocab(src_vocab)

    with open(args.tgt_vocab_pkl, "rb") as f:
        tgt_vocab = pickle.load(f)
    tgt_vocab = Vocab(tgt_vocab)

    src_tokenizer = get_tokenizer(args.src_tokenizer_name, language=args.src_tokenizer_language)
    tgt_tokenizer = get_tokenizer(args.tgt_tokenizer_name, language=args.tgt_tokenizer_language)

    collate_fn = CollateFunction(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        bos_idx=bos_idx,
        pad_idx=pad_idx,
    )

    train_dataset = Multi30k(split="train", language_pair=(args.src_language, args.tgt_language))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if platform.system() == "Windows" else os.cpu_count(),

        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )
    valid_dataset = Multi30k(split="valid", language_pair=(args.src_language, args.tgt_language))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    transformer = Seq2SeqTransformer(
        num_encoder_layers=3,
        num_decoder_layers=3,
        emb_size=512,
        num_head=8,
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        dim_feedforward=512,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = StepLR(
        optimizer=optimizer,
        step_size=args.lr_scheduler_step_size,
        gamma=args.lr_scheduler_gamma
    )

    trainer = Trainer(
        model=transformer,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )
    trainer.train()

    return


if __name__ == '__main__':
    main()
