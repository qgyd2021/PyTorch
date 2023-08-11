#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolbox.open_nmt.modules.decoders.decoder import Decoder


@Decoder.register("gru")
class GruDecoder(Decoder):
    def __init__(self,
                 max_decoding_length: int,
                 hidden_size: int,
                 vocab_size: int,
                 bos_idx: int,
                 device: torch.device = None,
                 ):
        super().__init__()
        self.max_decoding_length = max_decoding_length
        self.bos_idx = bos_idx
        self.device = device

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.projection_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self,
                memory: torch.Tensor,
                additions: Dict[str, Any],
                targets: torch.Tensor = None
                ):
        batch_size = memory.size(0)
        decoder_hidden_state = additions["hidden"]

        decoder_inputs = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.bos_idx)
        decoder_outputs = []

        max_decoding_length = targets.size(1) if targets is not None else self.max_decoding_length

        for i in range(max_decoding_length):
            decoder_output, decoder_hidden_state = self.forward_step(
                decoder_inputs, decoder_hidden_state, memory
            )
            decoder_outputs.append(decoder_output)

            if targets is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_inputs = targets[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_inputs = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        cache = [decoder_hidden_state]
        return decoder_outputs, cache

    def forward_step(self,
                     inputs: torch.Tensor,
                     hidden_state: torch.Tensor,
                     memory: torch.Tensor,
                     ):
        outputs = self.embedding(inputs)
        outputs = F.relu(outputs)
        outputs, hidden_state = self.gru(outputs, hidden_state)
        outputs = self.projection_layer(outputs)

        cache = [hidden_state]
        return outputs, cache


class BahdanauAttention(nn.Module):
    """
    https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


@Decoder.register("gru_attention")
class GruAttentionDecoder(Decoder):
    def __init__(self,
                 max_decoding_length: int,
                 hidden_size: int,
                 vocab_size: int,
                 bos_idx: int,
                 dropout_prob: float = 0.1,
                 device: torch.device = None,
                 ):
        super().__init__()
        self.max_decoding_length = max_decoding_length
        self.bos_idx = bos_idx
        self.device = device

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.attention = BahdanauAttention(hidden_size)

        self.gru = nn.GRU(
            2 * hidden_size,
            hidden_size,
            batch_first=True
        )
        self.projection_layer = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,
                memory: torch.Tensor,
                additions: Dict[str, Any],
                targets: torch.Tensor = None
                ):
        """
        :param memory: shape=[batch_size, seq_len, dim]
        :param additions: hidden state from GruEncoder,
        :param targets:
        :return:
        """
        batch_size = memory.size(0)

        decoder_hidden_state = additions["hidden"]

        decoder_inputs = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.bos_idx)

        decoder_outputs = []
        attentions = []

        max_decoding_length = targets.size(1) if targets is not None else self.max_decoding_length

        for i in range(max_decoding_length):
            decoder_output, [decoder_hidden_state, attn_weights] = self.forward_step(
                decoder_inputs, decoder_hidden_state, memory
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if targets is not None and targets.size(1) > i:
                # Teacher forcing: Feed the target as the next input
                decoder_inputs = targets[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_inputs = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        cache = [decoder_hidden_state, attentions]
        return decoder_outputs, cache

    def forward_step(self,
                     inputs: torch.Tensor,
                     hidden_state: torch.Tensor,
                     memory: torch.Tensor,
                     ):

        embedded = self.dropout(self.embedding(inputs))

        query = hidden_state.permute(1, 0, 2)
        context, attn_weights = self.attention(query, memory)
        input_gru = torch.cat((embedded, context), dim=2)

        outputs, hidden_state = self.gru(input_gru, hidden_state)
        outputs = self.projection_layer(outputs)

        cache = [hidden_state, attn_weights]
        return outputs, cache


if __name__ == '__main__':
    pass
