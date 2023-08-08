#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear

from toolbox.torchtext.nn import util
from toolbox.torchtext.modules.multi_head_self_attention import MultiHeadSelfAttention
from toolbox.torchtext.modules.feedforward import FeedForward
from toolbox.torchtext.modules.layer_norm import LayerNorm


class StackedSelfAttentionEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1
                 ) -> None:
        super(StackedSelfAttentionEncoder, self).__init__()

        self._use_positional_encoding = use_positional_encoding
        self._attention_layers: List[MultiHeadSelfAttention] = []
        self._feedfoward_layers: List[FeedForward] = []
        self._layer_norm_layers: List[LayerNorm] = []
        self._feed_forward_layer_norm_layers: List[LayerNorm] = []

        feedfoward_input_dim = input_dim
        for i in range(num_layers):
            feedfoward = FeedForward(feedfoward_input_dim,
                                     activations=['relu', 'linear'],
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     num_layers=2,
                                     dropout=dropout_prob)

            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedfoward_layers.append(feedfoward)

            feedforward_layer_norm = LayerNorm(feedfoward.get_output_dim())
            self.add_module(f"feedforward_layer_norm_{i}", feedforward_layer_norm)
            self._feed_forward_layer_norm_layers.append(feedforward_layer_norm)

            self_attention = MultiHeadSelfAttention(num_heads=num_attention_heads,
                                                    input_dim=hidden_dim,
                                                    attention_dim=projection_dim,
                                                    values_dim=projection_dim,
                                                    attention_dropout_prob=attention_dropout_prob)
            self.add_module(f"self_attention_{i}", self_attention)
            self._attention_layers.append(self_attention)

            layer_norm = LayerNorm(self_attention.get_output_dim())
            self.add_module(f"layer_norm_{i}", layer_norm)
            self._layer_norm_layers.append(layer_norm)

            feedfoward_input_dim = hidden_dim

        self.dropout = Dropout(residual_dropout_prob)
        self._input_dim = input_dim
        self._output_dim = self._attention_layers[-1].get_output_dim()

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def is_bidirectional(self):
        return False

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        if self._use_positional_encoding:
            output = util.add_positional_features(inputs)
        else:
            output = inputs
        for i in range(len(self._attention_layers)):
            attention = getattr(self, f"self_attention_{i}")
            feedforward = getattr(self, f"feedforward_{i}")
            feedforward_layer_norm = getattr(self, f"feedforward_layer_norm_{i}")
            layer_norm = getattr(self, f"layer_norm_{i}")
            cached_input = output
            feedforward_output = feedforward(output)
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                feedforward_output = feedforward_layer_norm(feedforward_output + cached_input)
            # shape (batch_size, sequence_length, hidden_dim)
            attention_output = attention(feedforward_output, mask)
            output = layer_norm(self.dropout(attention_output) + feedforward_output)

        return output


if __name__ == '__main__':
    pass
