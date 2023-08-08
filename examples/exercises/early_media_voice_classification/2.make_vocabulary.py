#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json

from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)

    parser.add_argument('--vocabulary_dir', default='./vocabulary', type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    vocabulary = Vocabulary()

    with open(args.train_subset, "r", encoding="utf-8") as f:
        for row in f:
            row = json.loads(row)
            label = row["label"]
            vocabulary.add_token_to_namespace(label, namespace="labels")

    vocabulary.save_to_files(args.vocabulary_dir)
    return


if __name__ == '__main__':
    main()
