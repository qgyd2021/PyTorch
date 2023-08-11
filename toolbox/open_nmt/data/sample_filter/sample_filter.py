#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import re
from typing import Any, Dict, List
import unicodedata

from toolbox.open_nmt.common.registrable import Registrable


class SampleFilter(Registrable):
    def filter(self, sample: Dict[str, Any]) -> bool:
        raise NotImplementedError


@SampleFilter.register("length")
class LengthFilter(SampleFilter):
    def __init__(self,
                 min_length: str = 0,
                 max_length: str = float("inf")):
        self.min_length = min_length
        self.max_length = max_length

    def filter(self, sample: Dict[str, Any]) -> bool:
        src: str = sample["src"]
        tgt: str = sample["tgt"]

        src_len = len(src.split(" "))
        tgt_len = len(tgt.split(" "))
        if self.min_length < src_len < self.max_length and self.min_length < tgt_len < self.max_length:
            return False
        return True


@SampleFilter.register("prefix")
class PrefixFilter(SampleFilter):
    def __init__(self,
                 src_prefixes: List[str] = None,
                 tgt_prefixes: List[str] = None,
                 ):
        self.src_prefixes = tuple(src_prefixes) if src_prefixes else None
        self.tgt_prefixes = tuple(tgt_prefixes) if tgt_prefixes else None

    def filter(self, sample: Dict[str, Any]) -> bool:
        src: str = sample["src"]
        tgt: str = sample["tgt"]

        if self.src_prefixes is not None and not src.startswith(self.src_prefixes):
            return True
        if self.tgt_prefixes is not None and not tgt.startswith(self.tgt_prefixes):
            return True
        return False


if __name__ == '__main__':
    pass
