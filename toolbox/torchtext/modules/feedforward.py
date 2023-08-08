#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear

from toolbox.torchtext.nn import util
from toolbox.torchtext.modules.activation import name2activation


class FeedForward(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, List[int]],
                 activations: Union[str, List[str]],
                 dropout: Union[float, List[float]] = 0.0) -> None:

        super(FeedForward, self).__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers
        if len(hidden_dims) != num_layers:
            raise AssertionError("len(hidden_dims) (%d) != num_layers (%d)" %
                                 (len(hidden_dims), num_layers))
        if len(activations) != num_layers:
            raise AssertionError("len(activations) (%d) != num_layers (%d)" %
                                 (len(activations), num_layers))
        if len(dropout) != num_layers:
            raise AssertionError("len(dropout) (%d) != num_layers (%d)" %
                                 (len(dropout), num_layers))
        self._activations = [name2activation[activation]() for activation in activations]

        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self.output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            output = dropout(activation(layer(output)))
        return output


def demo1():
    from toolbox.torchtext.nn import util

    embedding = nn.Embedding(7, 10)

    tokens: torch.LongTensor = torch.tensor(data=[[1, 2, 3, 0, 0], [2, 2, 3, 1, 0]], dtype=torch.long)
    mask = util.get_text_field_mask(tokens)

    inputs = embedding(tokens)

    feedforward = FeedForward(
        input_dim=10,
        num_layers=2,
        hidden_dims=10,
        activations=['relu', 'linear']
    )

    result = feedforward.forward(inputs)
    print(result.shape)

    return


if __name__ == '__main__':
    demo1()
