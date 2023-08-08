#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear

from toolbox.torchtext.nn import util


class MultiHeadSelfAttention(nn.Module):

    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        if values_dim % num_heads != 0:
            raise ValueError(f"Value size ({values_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        self._combined_projection = Linear(input_dim, 2 * attention_dim + values_dim)

        self._scale = (input_dim // num_heads) ** 0.5
        self._output_projection = Linear(values_dim, self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:

        num_heads = self._num_heads

        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new_ones(batch_size, timesteps)

        combined_projection = self._combined_projection(inputs)

        queries, keys, *values = combined_projection.split(self._attention_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()
        # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
        values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim/num_heads))
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * num_heads, timesteps, int(self._values_dim/num_heads))

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        queries_per_head = queries.view(batch_size, timesteps, num_heads, int(self._attention_dim/num_heads))
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim/num_heads))

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        keys_per_head = keys.view(batch_size, timesteps, num_heads, int(self._attention_dim/num_heads))
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim/num_heads))

        # shape (num_heads * batch_size, timesteps, timesteps)
        scaled_similarities = torch.bmm(queries_per_head / self._scale, keys_per_head.transpose(1, 2))

        # shape (num_heads * batch_size, timesteps, timesteps)
        attention = util.masked_softmax(scaled_similarities,
                                   mask.repeat(1, num_heads).view(batch_size * num_heads, timesteps),
                                   memory_efficient=True)
        attention = self._attention_dropout(attention)

        outputs = util.weighted_sum(values_per_head, attention)

        # Reshape back to original shape (batch_size, timesteps, values_dim)
        # shape (batch_size, num_heads, timesteps, values_dim/num_heads)
        outputs = outputs.view(batch_size, num_heads, timesteps, int(self._values_dim / num_heads))
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, self._values_dim)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        return outputs


def demo1():
    from toolbox.torchtext.nn import util

    embedding = nn.Embedding(7, 10)

    tokens: torch.LongTensor = torch.tensor(data=[[1, 2, 3, 0, 0], [2, 2, 3, 1, 0]], dtype=torch.long)
    mask = util.get_text_field_mask(tokens)

    inputs = embedding(tokens)

    attention = MultiHeadSelfAttention(
        num_heads=2,
        input_dim=10,
        attention_dim=10,
        values_dim=10,
        output_projection_dim=10
    )

    result = attention.forward(inputs, mask)
    print(result.shape)

    return


if __name__ == '__main__':
    demo1()
