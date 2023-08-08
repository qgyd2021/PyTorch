#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear


class LayerNorm(nn.Module):
    def __init__(self,
                 dimension: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dimension))
        self.beta = torch.nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, tensor: torch.Tensor):
        mean = tensor.mean(-1, keepdim=True)
        std = tensor.std(-1, unbiased=False, keepdim=True)
        return self.gamma * (tensor - mean) / (std + self.eps) + self.beta


def demo1():
    from toolbox.torchtext.nn import util

    embedding = nn.Embedding(7, 10)

    tokens: torch.LongTensor = torch.tensor(data=[[1, 2, 3, 0, 0], [2, 2, 3, 1, 0]], dtype=torch.long)
    mask = util.get_text_field_mask(tokens)

    inputs = embedding(tokens)
    print(inputs.shape)

    layer_norm = LayerNorm(
        dimension=10,
    )

    result = layer_norm.forward(inputs)
    print(result.shape)

    print(layer_norm.gamma.shape)
    return


if __name__ == '__main__':
    demo1()
