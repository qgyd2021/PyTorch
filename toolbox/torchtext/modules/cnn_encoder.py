#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear

from toolbox.torchtext.nn import util
from toolbox.torchtext.modules.activation import name2activation


class CnnEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_filters: int,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
                 conv_layer_activation: str = 'relu',
                 output_dim: Optional[int] = None) -> None:
        super(CnnEncoder, self).__init__()
        self._input_dim = input_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = name2activation[conv_layer_activation]()
        self._output_dim = output_dim

        self._convolution_layers = [nn.Conv1d(in_channels=self._input_dim,
                                           out_channels=self._num_filters,
                                           kernel_size=ngram_size)
                                    for ngram_size in self._ngram_filter_sizes]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        tokens = torch.transpose(tokens, 1, 2)

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(
                    self._activation(convolution_layer(tokens)).max(dim=2)[0]
            )

        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result


def demo1():
    from toolbox.torchtext.nn import util

    embedding = nn.Embedding(7, 10)

    tokens: torch.LongTensor = torch.tensor(data=[[1, 2, 3, 0, 0], [2, 2, 3, 1, 0]], dtype=torch.long)
    mask = util.get_text_field_mask(tokens)

    inputs = embedding(tokens)

    cnn = CnnEncoder(
        input_dim=10,
        num_filters=3,
        ngram_filter_sizes=(2, 3, 4, 5),
        # output_dim=10,
    )

    result = cnn.forward(inputs, mask)
    print(result.shape)

    return


if __name__ == '__main__':
    demo1()
