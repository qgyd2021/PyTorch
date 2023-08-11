#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict

import torch
import torch.nn as nn

from toolbox.open_nmt.models.model import Model
from toolbox.open_nmt.modules.encoders.encoder import Encoder
from toolbox.open_nmt.modules.decoders.decoder import Decoder
from toolbox.open_nmt.nn import util


@Model.register("seq2seq")
class Seq2SeqTranslation(Model):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder
                 ):
        super(Seq2SeqTranslation, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if src_mask is None:
            src_mask = util.get_text_field_mask(src_ids).float()
        if tgt_mask is None:
            tgt_mask = util.get_text_field_mask(tgt_ids).float()

        memory, additions = self.encoder.forward(src_ids, src_mask)
        logits, _ = self.decoder.forward(memory, additions, targets=tgt_ids)
        result = {
            "logits": logits
        }
        return result

    def greedy_decode(self,
                      src_ids: torch.Tensor,
                      src_mask: torch.Tensor = None,
                      ):
        """
        :param src_ids: shape=[batch_size, seq_len]
        :param src_mask: shape=[batch_size, seq_len]
        :return:
        """
        if src_mask is None:
            src_mask = util.get_text_field_mask(src_ids).float()

        memory, additions = self.encoder.forward(src_ids, src_mask)
        logits, _ = self.decoder.forward(memory, additions, targets=None)

        probs = torch.softmax(logits, dim=-1)
        hypothesis = torch.argmax(probs, dim=-1)
        hypothesis = hypothesis.numpy().tolist()

        return hypothesis


if __name__ == '__main__':
    pass
