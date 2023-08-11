#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as f

from toolbox.open_nmt.common.registrable import Registrable


class Decoder(nn.Module, Registrable):
    def forward(self,
                memory: torch.Tensor,
                additions: Dict[str, Any],
                targets: torch.LongTensor = None
                ):
        """

        :param memory: encoder outputs, shape=[batch_size, seq_len, dim]
        :param additions:
        :param targets: shape=[batch_size, seq_len]
        :return:
        outputs: shape=[batch_size, seq_len, token]
        cache: list of tensor.
        """
        raise NotImplementedError

    def forward_step(self,
                     inputs: torch.Tensor,
                     hidden_state: torch.Tensor,
                     memory: torch.Tensor,
                     ):
        """
        :param inputs:
        :param hidden_state:
        :param memory:
        :return:
        outputs: shape=[batch_size, seq_len, token]
        cache: list of tensor.
        """
        raise NotImplementedError


if __name__ == '__main__':
    pass
