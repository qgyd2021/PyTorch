#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from project_settings import project_path
from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_dir",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )

    parser.add_argument("--vocabulary_dir", default="./vocabulary", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    vocabulary = Vocabulary()

    class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    for class_name in class_names:
        vocabulary.add_token_to_namespace(class_name, namespace="labels")

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
