#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import Counter
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import load_dataset, DatasetDict
import huggingface_hub
import numpy as np
from tqdm import tqdm

import project_settings as settings
from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    # https://huggingface.co/datasets/NebulaByte/E-Commerce_FAQs
    # https://huggingface.co/datasets/NebulaByte/E-Commerce_Customer_Support_Conversations
    parser.add_argument("--dataset_path", default="bitext/customer-support-intent-dataset", type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--dataset_split", default=None, choices=["train", "test", "validation", "all"], type=str)
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

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    vocabulary = Vocabulary()

    dataset: DatasetDict = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir
    )
    print(dataset)

    counter = Counter()
    for sample in tqdm(dataset["train"], desc="train subset"):
        text = sample["utterance"]
        label = sample["intent"]
        vocabulary.add_token_to_namespace(label, namespace="labels")
        counter.update([label])

    for sample in tqdm(dataset["validation"], desc="validation subset"):
        text = sample["utterance"]
        label = sample["intent"]
        vocabulary.add_token_to_namespace(label, namespace="labels")
        counter.update([label])

    print(counter.most_common(n=100))

    vocabulary.set_from_file(
        filename=os.path.join(args.pretrained_model_dir, "vocab.txt"),
        is_padded=False,
        oov_token="[UNK]",
        namespace="tokens",
    )

    vocabulary.save_to_files(args.vocabulary_dir)

    return


if __name__ == '__main__':
    main()
