#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import Counter
import pickle
from typing import Any, Dict, Iterable, List

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator, Vocab

from toolbox.torch.utils.data.vocabulary import Vocabulary


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

    parser.add_argument("--vocabulary_dir", default="./vocabulary", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    multi30k.URL["train"] = args.multi30k_train_url
    multi30k.URL["valid"] = args.multi30k_valid_url

    src_tokenizer = get_tokenizer(args.src_tokenizer_name, language=args.src_tokenizer_language)
    tgt_tokenizer = get_tokenizer(args.tgt_tokenizer_name, language=args.tgt_tokenizer_language)

    unk_idx = 0
    pad_idx = 1
    bos_idx = 2
    eos_idx = 3
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

    train_dataset_iter = Multi30k(
        split="train",
        language_pair=(
            args.src_language,
            args.tgt_language,
        )
    )

    src_token_counter = Counter()
    tgt_token_counter = Counter()
    for sample in train_dataset_iter:
        src_sample, tgt_sample = sample
        src_tokens = src_tokenizer(src_sample)
        tgt_tokens = tgt_tokenizer(tgt_sample)
        src_token_counter.update(src_tokens)
        tgt_token_counter.update(tgt_tokens)

    src_token_counter = list(sorted(src_token_counter.items(), key=lambda x: x[1], reverse=True))
    tgt_token_counter = list(sorted(tgt_token_counter.items(), key=lambda x: x[1], reverse=True))

    vocabulary = Vocabulary(
        non_padded_namespaces=["src_tokens", "tgt_tokens"],
        padding_token="<pad>",
        oov_token="<unk>"
    )
    for special_symbol in special_symbols:
        vocabulary.add_token_to_namespace(special_symbol, namespace="src_tokens")

    for token, freq in src_token_counter:
        vocabulary.add_token_to_namespace(token, namespace="src_tokens")

    for special_symbol in special_symbols:
        vocabulary.add_token_to_namespace(special_symbol, namespace="tgt_tokens")
    for token, freq in tgt_token_counter:
        vocabulary.add_token_to_namespace(token, namespace="tgt_tokens")

    vocabulary.save_to_files(args.vocabulary_dir)

    return


if __name__ == '__main__':
    main()
