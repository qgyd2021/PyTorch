#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as f

from toolbox.open_nmt.common.registrable import Registrable


class Encoder(nn.Module, Registrable):

    def forward(self,
                inputs: torch.LongTensor,
                mask: torch.Tensor,
                ):
        """
        :param inputs: shape=[batch_size, seq_len]
        :param mask:
        :return:
        """
        raise NotImplementedError
