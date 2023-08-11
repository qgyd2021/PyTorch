#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import Counter
from typing import Dict, Iterable, Optional, Set, Tuple

import torch


def ngrams(
    tensor: torch.LongTensor, ngram_size: int, exclude_indices: Set[int]
) -> Dict[Tuple[int, ...], int]:
    ngram_counts: Dict[Tuple[int, ...], int] = Counter()
    if ngram_size > tensor.size(-1):
        return ngram_counts
    for start_position in range(ngram_size):
        for tensor_slice in tensor[start_position:].split(ngram_size, dim=-1):
            if tensor_slice.size(-1) < ngram_size:
                break
            ngram = tuple(x.item() for x in tensor_slice)
            if any(x in exclude_indices for x in ngram):
                continue
            ngram_counts[ngram] += 1
    return ngram_counts


def get_valid_tokens_mask(tensor: torch.LongTensor, exclude_indices: Set[int]) -> torch.ByteTensor:
    valid_tokens_mask = torch.ones_like(tensor, dtype=torch.bool)
    for index in exclude_indices:
        valid_tokens_mask &= tensor != index
    return valid_tokens_mask
