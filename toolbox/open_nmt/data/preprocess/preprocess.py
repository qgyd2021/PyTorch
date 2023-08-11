#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import re
from typing import Any, Dict, List
import unicodedata

from toolbox.open_nmt.common.registrable import Registrable


class Preprocess(Registrable):
    def process(self, sample: Dict[str, Any]):
        raise NotImplementedError


@Preprocess.register("lowercase")
class LowercasePreprocess(Preprocess):
    def process(self, sample: Dict[str, Any]) -> Dict[str, str]:
        src = sample["src"]
        tgt = sample["tgt"]

        result = {
            "src": src.lower(),
            "tgt": tgt.lower()
        }
        return result


@Preprocess.register("reverse")
class ReversePreprocess(Preprocess):
    def process(self, sample: Dict[str, Any]) -> Dict[str, str]:
        src = sample["src"]
        tgt = sample["tgt"]

        result = {
            "src": tgt,
            "tgt": src
        }
        return result


@Preprocess.register("unicode_to_ascii")
class UnicodeToAsciiPreprocess(Preprocess):
    @staticmethod
    def unicode_to_ascii(text: str) -> str:
        text = "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
        text = re.sub(r"([.!?])", r" \1", text)
        text = re.sub(r"[^a-zA-Z!?]+", r" ", text)
        text = text.strip()
        return text

    def process(self, sample: Dict[str, Any]) -> Dict[str, str]:
        src = sample["src"]
        tgt = sample["tgt"]
        src = self.unicode_to_ascii(src)
        tgt = self.unicode_to_ascii(tgt)
        result = {
            "src": src,
            "tgt": tgt
        }
        return result


if __name__ == '__main__':
    pass
