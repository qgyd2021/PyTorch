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

from toolbox.open_nmt.data.collate_functions.collate_fn import BasicCollateFunction
from toolbox.open_nmt.data.tokenizers.tokenizer import WhitespaceTokenizer
from toolbox.open_nmt.data import vocabulary
from toolbox.open_nmt.modules.encoders.gru_encoder import GRUEncoder
from toolbox.open_nmt.modules.decoders.gru_decoder import GruAttentionDecoder
from toolbox.open_nmt.models.seq2seq_translation import Seq2SeqTranslation
from toolbox.open_nmt.predictors.predictor import BasicPredictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabulary_dir", default="./vocabulary", type=str)
    parser.add_argument("--weights_file", default="serialization_dir/model_state_79.th", type=str)

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
    with open(args.weights_file, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")

    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()

    collate_fn = BasicCollateFunction(
        src_tokenizer=WhitespaceTokenizer(),
        tgt_tokenizer=WhitespaceTokenizer(),
        vocabulary=vocab,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
    )

    predictor = BasicPredictor(
        model=model,
        collate_fn=collate_fn,
        vocabulary=vocab,
        device=device
    )

    sample = {
        "src": "Elle est en train de l'aider.",
        "tgt": "She is helping him.	Elle est en train de l'aider."
    }

    # batch = collate_fn.__call__(batch_sample=[sample])
    # inputs, targets = batch
    # hypothesis = model.greedy_decode(src_ids=inputs)

    result = predictor.predict_json(inputs=sample, mode="greedy_decode")
    print(result)
    return


if __name__ == '__main__':
    main()
