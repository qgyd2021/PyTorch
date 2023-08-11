#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from toolbox.open_nmt.modules.encoders.encoder import Encoder


@Encoder.register("gru")
class GRUEncoder(Encoder):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 ):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.embedding_dropout = nn.Dropout(dropout)

    def forward(self,
                inputs: torch.LongTensor,
                mask: torch.Tensor,
                ):
        embedded = self.embedding_dropout(self.embedding(inputs))
        outputs, hidden = self.gru(embedded)
        additions = {
            "hidden": hidden
        }
        return outputs, additions


if __name__ == '__main__':
    pass
