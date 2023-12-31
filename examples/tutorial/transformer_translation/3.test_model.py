#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import math
import os
import pickle
import platform
import random
import sys
from typing import Any, Callable, Dict, Iterable, List, Union

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

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
from torchtext._torchtext import Vocab as VocabPybind
from tqdm import tqdm

from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    """
    python3 3.test_model.py --weights_file data_dir/serialization_dir/model_state_9.th --vocabulary_dir data_dir/vocabulary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file", default="serialization_dir/model_state_1.th", type=str)

    parser.add_argument("--src_language", default="de", type=str)
    parser.add_argument("--tgt_language", default="en", type=str)

    parser.add_argument("--src_tokenizer_name", default="spacy", type=str)
    parser.add_argument("--src_tokenizer_language", default="de_core_news_sm", type=str)
    parser.add_argument("--tgt_tokenizer_name", default="spacy", type=str)
    parser.add_argument("--tgt_tokenizer_language", default="en_core_web_sm", type=str)

    parser.add_argument("--vocabulary_dir", default="./vocabulary", type=str)

    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--device_id", default=0, type=int)

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


class CollateFunction(object):
    def __init__(self,
                 src_tokenizer,
                 tgt_tokenizer,
                 src_vocab: Vocab,
                 tgt_vocab: Vocab,
                 bos_idx: int,
                 eos_idx: int,
                 pad_idx: int,
                 ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
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
                    torch.tensor([self.eos_idx])
                )
            )

            tgt_tokens = self.tgt_tokenizer(tgt_sample)
            tgt_token_ids = self.tgt_vocab(tgt_tokens)
            tgt_input_ids = torch.cat(
                tensors=(
                    torch.tensor([self.bos_idx]),
                    torch.tensor(tgt_token_ids),
                    torch.tensor([self.eos_idx])
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


def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_idx, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)

        seq_len = ys.size(0)
        mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        tgt_mask = mask.type(torch.bool).to(device)

        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            tensors=[
                ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)
            ],
            dim=0
        )
        if next_word == eos_idx:
            break
    return ys


def main():
    args = get_args()

    device = "cpu"

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    unk_idx = 0
    pad_idx = 1
    bos_idx = 2
    eos_idx = 3
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

    src_token_to_index = vocabulary.get_token_to_index_vocabulary(namespace="src_tokens")
    src_token_to_index = list(sorted(src_token_to_index.items(), key=lambda x: x[1], reverse=False))
    src_tokens = [k for k, v in src_token_to_index]
    src_vocab = Vocab(VocabPybind(src_tokens, None))
    src_vocab.set_default_index(unk_idx)

    tgt_token_to_index = vocabulary.get_token_to_index_vocabulary(namespace="tgt_tokens")
    tgt_token_to_index = list(sorted(tgt_token_to_index.items(), key=lambda x: x[1], reverse=False))
    tgt_tokens = [k for k, v in tgt_token_to_index]
    tgt_vocab = Vocab(VocabPybind(tgt_tokens, None))
    tgt_vocab.set_default_index(unk_idx)

    src_tokenizer = get_tokenizer(args.src_tokenizer_name, language=args.src_tokenizer_language)
    tgt_tokenizer = get_tokenizer(args.tgt_tokenizer_name, language=args.tgt_tokenizer_language)

    collate_fn = CollateFunction(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
    )

    model = Seq2SeqTransformer(
        num_encoder_layers=3,
        num_decoder_layers=3,
        emb_size=512,
        num_head=8,
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        dim_feedforward=512,
    )

    with open(args.weights_file, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")

    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()

    sample = {
        "src": "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.",
        "tgt": "Two young, White males are outside near many bushes.",
    }
    inputs, _ = collate_fn.__call__(batch=[sample])

    src = inputs["src"]
    src_mask = inputs["src_mask"]

    num_tokens = src.shape[0]

    tgt_tokens = greedy_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_len=num_tokens + 5,
        start_symbol=bos_idx,
        eos_idx=eos_idx,
        device=device,
    ).flatten()

    tgt = " ".join(tgt_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

    print(tgt)
    return


if __name__ == '__main__':
    main()
