#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from toolbox.open_nmt.common.registrable import Registrable
from toolbox.open_nmt.data.tokenizers.tokenizer import Tokenizer
from toolbox.open_nmt.data.vocabulary import Vocabulary


class CollateFunction(Registrable):
    def __init__(self):
        self.training = False

    def __call__(self, batch_sample: List[dict]):
        raise NotImplementedError

    def train(self):
        self.training = True
        return self.training

    def eval(self):
        self.training = False
        return self.training


@CollateFunction.register("basic")
class BasicCollateFunction(CollateFunction):
    def __init__(self,
                 src_tokenizer: Tokenizer,
                 tgt_tokenizer: Tokenizer,
                 vocabulary: Vocabulary,
                 pad_idx: int,
                 bos_idx: int,
                 eos_idx: int,
                 ):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.vocabulary = vocabulary

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __call__(self, batch_sample: List[dict]):
        src_ids_list = list()
        tgt_ids_list = list()
        for sample in batch_sample:
            src = sample["src"]
            tgt = sample["tgt"]
            src_tokens = self.src_tokenizer.tokenize(src)
            tgt_tokens = self.tgt_tokenizer.tokenize(tgt)

            src_ids = [
                self.vocabulary.get_token_index(token=token, namespace="src_tokens")
                for token in src_tokens
            ]
            tgt_ids = [
                self.vocabulary.get_token_index(token=token, namespace="tgt_tokens")
                for token in tgt_tokens
            ]
            src_ids = [self.bos_idx] + src_ids + [self.eos_idx]
            tgt_ids = [self.bos_idx] + tgt_ids + [self.eos_idx]

            src_ids = torch.tensor(src_ids)
            tgt_ids = torch.tensor(tgt_ids)

            src_ids_list.append(src_ids)
            tgt_ids_list.append(tgt_ids)

        # [seq_len, batch_size]
        src_batch = pad_sequence(src_ids_list, padding_value=self.pad_idx)
        tgt_batch = pad_sequence(tgt_ids_list, padding_value=self.pad_idx)

        # [batch_size, seq_len]
        src_batch = src_batch.transpose(0, 1).contiguous().detach()
        tgt_batch = tgt_batch.transpose(0, 1).contiguous().detach()
        return src_batch, tgt_batch
