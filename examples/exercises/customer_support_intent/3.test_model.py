#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""
import argparse
from typing import List

import numpy as np
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizer

from project_settings import project_path
from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="i didnt feel humiliated", type=str)

    parser.add_argument(
        "--pretrained_model_dir",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    parser.add_argument("--vocabulary_dir", default="vocabulary", type=str)
    parser.add_argument("--weights_file", default="serialization_dir/best.th", type=str)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--device_id", default=0, type=int)

    args = parser.parse_args()
    return args


class CollateFunction(object):
    def __init__(self,
                 vocabulary: Vocabulary,
                 tokenizer: BertTokenizer,
                 text_field: str = "text",
                 label_field: str = "label"
                 ):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.label_field = label_field

    def __call__(self, batch: List[dict]):
        texts: List[str] = list()
        targets: List[int] = list()
        for example in batch:
            text = example[self.text_field]
            label = example[self.label_field]
            texts.append(text)

            label_idx = self.vocabulary.get_token_index(token=label)
            targets.append(label_idx)

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

    device = torch.device("{}:{}".format(args.device, args.device_id))
    n_gpu = torch.cuda.device_count()

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)

    collate_fn = CollateFunction(
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        text_field="utterance",
        label_field="intent"
    )

    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_dir,
        num_labels=vocabulary.get_vocab_size(namespace="labels")
    )
    model.to(device)

    with open(args.weights_file, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")

    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()

    example = {
        "utterance": args.text,
        "intent": "unknown"
    }

    inputs, targets = collate_fn.__call__([example])
    inputs = inputs.to(device)

    with torch.no_grad():
        seq_cls_output: SequenceClassifierOutput = model.forward(inputs)
        logits = seq_cls_output.logits
        probs = torch.softmax(logits, dim=-1)
        label_idx = torch.argmax(probs, dim=-1)
        label_idx = label_idx.detach().cpu().numpy().tolist()
        probs = probs.numpy().tolist()
        for l_idx, pbs in zip(label_idx, probs):
            label_str = vocabulary.get_token_from_index(index=l_idx, namespace="labels")
            prob = pbs[l_idx]
            print("label: {}, prob: {}".format(label_str, round(prob, 4)))

    return


if __name__ == '__main__':
    main()
