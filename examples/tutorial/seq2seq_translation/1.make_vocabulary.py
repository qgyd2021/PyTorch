#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import Counter
import re
from typing import Any, Dict, List
import unicodedata

from torch.utils.data.dataset import Dataset

from toolbox.open_nmt.data.dataset.dataset import PairDataset
from toolbox.open_nmt.data.preprocess.preprocess import LowercasePreprocess, ReversePreprocess, UnicodeToAsciiPreprocess
from toolbox.open_nmt.data.sample_filter.sample_filter import LengthFilter, PrefixFilter
from toolbox.open_nmt.data.dataset.preprocess_filter_dataset import PreprocessFilterDataset
from toolbox.open_nmt.data.tokenizers.tokenizer import WhitespaceTokenizer
from toolbox.open_nmt.data import vocabulary


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pair_file", default="data_dir/data/eng-fra.txt", type=str)

    parser.add_argument("--vocabulary_dir", default="./vocabulary", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

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

    tokenizer = WhitespaceTokenizer()

    src_counter = Counter()
    tgt_counter = Counter()
    for sample in dataset:
        src = sample["src"]
        tgt = sample["tgt"]
        src_tokens = tokenizer.tokenize(src)
        tgt_tokens = tokenizer.tokenize(tgt)

        src_counter.update(src_tokens)
        tgt_counter.update(tgt_tokens)

    src_counter = list(sorted(src_counter.items(), key=lambda x: x[1], reverse=True))
    tgt_counter = list(sorted(tgt_counter.items(), key=lambda x: x[1], reverse=True))

    pad_idx = 0
    bos_idx = 1
    eos_idx = 2
    unk_idx = 3
    special_symbols = ["<pad>", "<bos>", "<eos>", "<unk>"]

    vocabulary.DEFAULT_PADDING_TOKEN = "<pad>"
    vocabulary.DEFAULT_OOV_TOKEN = "<unk>"
    vocab = vocabulary.Vocabulary(
        non_padded_namespaces=["src_tokens", "tgt_tokens"],
    )
    for special_symbol in special_symbols:
        vocab.add_token_to_namespace(special_symbol, namespace="src_tokens")

    for token, freq in src_counter:
        vocab.add_token_to_namespace(token, namespace="src_tokens")

    for special_symbol in special_symbols:
        vocab.add_token_to_namespace(special_symbol, namespace="tgt_tokens")
    for token, freq in tgt_counter:
        vocab.add_token_to_namespace(token, namespace="tgt_tokens")

    vocab.save_to_files(args.vocabulary_dir)

    return


if __name__ == '__main__':
    main()
