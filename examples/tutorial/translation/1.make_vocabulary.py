#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""
import argparse
import pickle
from typing import Any, Dict, Iterable, List

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator, Vocab


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

    args = parser.parse_args()
    return args


def multi30k_text_to_tokens(
    data_iter: Iterable,
    language: str,
    language_index: Dict[str, int],
    language_to_tokenizer: Dict[str, Any]
) -> List[str]:
    for data_sample in data_iter:
        yield language_to_tokenizer[language](data_sample[language_index[language]])


def main():
    args = get_args()

    multi30k.URL["train"] = args.multi30k_train_url
    multi30k.URL["valid"] = args.multi30k_valid_url

    language_to_tokenizer = {
        args.src_language: get_tokenizer(args.src_tokenizer_name, language=args.src_tokenizer_language),
        args.tgt_language: get_tokenizer(args.tgt_tokenizer_name, language=args.tgt_tokenizer_language),
    }

    unk_idx = 0
    pad_idx = 1
    bos_idx = 2
    eos_idx = 3
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

    # src
    train_dataset_iter = Multi30k(
        split="train",
        language_pair=(
            args.src_language,
            args.tgt_language,
        )
    )

    vocab: Vocab = build_vocab_from_iterator(
        multi30k_text_to_tokens(
            train_dataset_iter,
            language=args.src_language,
            language_index={
                args.src_language: 0,
                args.tgt_language: 1,
            },
            language_to_tokenizer=language_to_tokenizer
        ),
        min_freq=1,
        specials=special_symbols,
        special_first=True,
    )

    vocab.set_default_index(unk_idx)

    with open(args.src_vocab_pkl, "wb") as f:
        pickle.dump(vocab.vocab, f)

    # tgt
    train_dataset_iter = Multi30k(
        split="train",
        language_pair=(
            args.src_language,
            args.tgt_language,
        )
    )

    vocab: Vocab = build_vocab_from_iterator(
        multi30k_text_to_tokens(
            train_dataset_iter,
            language=args.tgt_language,
            language_index={
                args.src_language: 0,
                args.tgt_language: 1,
            },
            language_to_tokenizer=language_to_tokenizer
        ),
        min_freq=1,
        specials=special_symbols,
        special_first=True,
    )

    vocab.set_default_index(unk_idx)

    with open(args.tgt_vocab_pkl, "wb") as f:
        pickle.dump(vocab.vocab, f)

    return


if __name__ == '__main__':
    main()
