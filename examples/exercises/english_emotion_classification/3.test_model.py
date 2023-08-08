#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""
import argparse
from typing import List

import numpy as np
import torch
from transformers.models.bert.tokenization_bert import BertTokenizer

from project_settings import project_path
from toolbox.torch.utils.data.vocabulary import Vocabulary
from toolbox.torchtext.models.text_classification.text_cnn import TextCNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="i didnt feel humiliated", type=str)

    parser.add_argument("--vocabulary_dir", default="vocabulary", type=str)
    parser.add_argument("--weights_file", default="serialization_dir/best.th", type=str)

    parser.add_argument(
        "--pretrained_model_dir",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )

    args = parser.parse_args()
    return args


class CollateFunction(object):
    def __init__(self, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[dict]):
        texts: List[str] = list()
        targets: List[int] = list()
        for example in batch:
            text = example["text"]
            label = example["label"]
            texts.append(text)
            targets.append(label)

        encodings = self.tokenizer.__call__(
            text=texts,
            padding="longest",
            truncation=True,
            max_length=512,
        )
        inputs = encodings["input_ids"]

        inputs = np.array(inputs)
        targets = np.array(targets)

        inputs: torch.LongTensor = torch.from_numpy(inputs)
        targets: torch.LongTensor = torch.from_numpy(targets)
        return inputs, targets


def main():
    args = get_args()

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)

    collate_fn = CollateFunction(
        tokenizer=tokenizer,
    )

    model = TextCNN(
        num_embeddings=vocabulary.get_vocab_size(namespace='tokens'),
        embedding_dim=128,
        stacked_self_attention_encoder_param={
            'input_dim': 128,
            'hidden_dim': 128,
            'projection_dim': 128,
            'feedforward_hidden_dim': 128,
            'num_layers': 2,
            'num_attention_heads': 4,
            'use_positional_encoding': False,
        },
        cnn_encoder_param={
            'input_dim': 128,
            'num_filters': 32,
            'ngram_filter_sizes': [2, 3, 4, 5],
            'output_dim': 128,
        },
        output_dim=128,
    )

    with open(args.weights_file, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()

    example = {
        "text": args.text,
        "label": 0
    }

    inputs, targets = collate_fn.__call__([example])

    with torch.no_grad():
        logits = model.forward(inputs)
        probs = torch.softmax(logits, dim=-1)
        label_idx = torch.argmax(probs, dim=-1)
        label_idx = label_idx.numpy().tolist()
        probs = probs.numpy().tolist()
        for l_idx, pbs in zip(label_idx, probs):
            label_str = vocabulary.get_token_from_index(index=l_idx, namespace="labels")
            prob = pbs[l_idx]
            print("label: {}, prob: {}".format(label_str, round(prob, 4)))

    return


if __name__ == '__main__':
    main()
