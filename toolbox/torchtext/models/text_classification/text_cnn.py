#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear

from toolbox.torchtext.nn import util
from toolbox.torchtext.modules.stacked_self_attention import StackedSelfAttentionEncoder
from toolbox.torchtext.modules.cnn_encoder import CnnEncoder


name2activation = {
    'linear': lambda: lambda x: x,
    'relu': nn.ReLU,
}


class TextCNN(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 stacked_self_attention_encoder_param: dict,
                 cnn_encoder_param: dict,
                 output_dim: int = None,
                 ):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.stacked_self_attention_encoder = StackedSelfAttentionEncoder(
            **stacked_self_attention_encoder_param
        )

        self.cnn_encoder = CnnEncoder(
            **cnn_encoder_param
        )

        if output_dim is not None:
            self.projection_layer = Linear(self.cnn_encoder.get_output_dim(), output_dim)
        else:
            self.projection_layer = None

    def forward(self, tokens: torch.LongTensor, mask: torch.LongTensor = None):
        """
        :param tokens: shape = [batch_size, seq_length]
        :param mask:
        :return:
        """
        x = self.embedding(tokens)
        if mask is None:
            mask = util.get_text_field_mask(tokens).float()

        x = x * mask.unsqueeze(-1).float()

        x = self.stacked_self_attention_encoder.forward(x, mask)
        x = self.cnn_encoder.forward(x, mask)

        if self.projection_layer:
            x = self.projection_layer.forward(x)

        return x


if __name__ == '__main__':
    pass
