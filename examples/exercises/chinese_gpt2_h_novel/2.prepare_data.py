#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from tqdm import tqdm
from transformers.models.bert.tokenization_bert import BertTokenizer

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=(project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix(),
        type=str
    )

    parser.add_argument("--dataset_path", default="qgyd2021/h_novel", type=str)
    # parser.add_argument("--dataset_name", default="ltxsba_500m", type=str)
    parser.add_argument("--dataset_name", default="ltxsba_5gb", type=str)
    parser.add_argument("--dataset_split", default="train", type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)

    parser.add_argument("--min_text_length", default=512, type=int)

    args = parser.parse_args()
    return args


class TextNormalization(object):
    """
    https://blog.csdn.net/buluxianfeng/article/details/126223346
    """

    def __init__(self):
        self.punctuation_map = {
            # "，": ",",
            # "。": ".",
            # "、": ",",
            # "？": "?",
            # "：": ":",
            # "｛": "{",
            # "｝": "}",
            # "（": "(",
            # "）": ")",
            # "【": "(",
            # "】": ")",
            # "「": "\"",
            # "」": "\"",
            "┏": "",
            "┓": "",
            "━": "",
            "◢": "",
            "◣": "",
            "┃": "|",
            "┗": "",
            "┛": "",
            # "『": "\"",
            # "』": "\"",
            # "《": "(",
            # "》": ")",
            "”": "\"",
            "“": "\"",
            "‘": "\'",
            "’": "\'",
            # "=": "",
            "—": "-",

            "": "",
            " ": "",
            " ": "",
            "\t": "",
            "\n": "",
            "\r": "",
            "\v": "",
            "\f": "",

            "…": "",

        }

        self.string_map = {
            "^_^": "",
            "◆": "",
            "☆": "",

            "()": "",
            "｛｝": "",
            "『』": "",
            "《》": "",

            "&lt;": "<",
            "&gt;": ">",
            "&amp;": " ",
            "&nbsp;": " ",
            "quot;": "\"",
            "nbsp;": " ",
            "amp;": "",
            "&#10008": "✘",

            "龙腾小说网ltxs520.com": "",
            "龙腾小说吧www.ltxsba.net": "",
            "龙腾小说网 ltxs520.com": "",
            "龙腾小说 ltxs520.com": "",
            "龙腾小说ltxs520.com": "",

            "(..)免费": "",

            "\"\"": "\"\n\"",

            "脔": "",
            "氵朝": "潮",
            "咕哝": "嘟囔",
            "迳自": "径自",
            "窸窣": "稀疏",

        }

    def is_q_number(self, uchar):
        """判断一个unicode是否是全角数字"""
        if u'\uff10' <= uchar <= u'\uff19':
            return True
        else:
            return False

    def is_q_alphabet(self, uchar):
        """判断一个unicode是否是全角英文字母"""
        if (u'\uff21' <= uchar <= u'\uff3a') or (u'\uff41' <= uchar <= u'\uff5a'):
            return True
        else:
            return False

    def q_to_b(self, uchar):
        """单个字符 全角转半角"""
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            return uchar
        return chr(inside_code)

    def number_alphabet_q_to_b(self, text: str):
        result = ""
        for c in text:
            if self.is_q_alphabet(c) or self.is_q_number(c):
                c = self.q_to_b(c)
            result += c
        return result

    def lowercase(self, text: str):
        result = str(text).lower()
        return result

    def replace_punctuation(self, text: str):
        text_ = text
        for k, v in self.punctuation_map.items():
            text_ = text_.replace(k, v)

        if text_ != text:
            text_ = self.replace_punctuation(text_)
        return text_

    def replace_string(self, text: str):
        text_ = text
        for k, v in self.string_map.items():
            text_ = text_.replace(k, v)

        if text_ != text:
            text_ = self.replace_string(text_)
        return text_

    def normalize(self, text: str):
        text = self.lowercase(text)
        # text = self.number_alphabet_q_to_b(text)
        text = self.replace_punctuation(text)
        text = self.replace_string(text)

        return text


def main():
    args = get_args()

    dataset_dict = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        # split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
        streaming=True,
    )

    train_dataset = dataset_dict["train"]

    text_ = ""
    with open(args.train_subset, "w", encoding="utf-8") as ftrain, \
        open(args.valid_subset, "w", encoding="utf-8") as fvalid:
        for sample in tqdm(train_dataset):

            source = sample["source"]
            idx = sample["idx"]
            filename = sample["filename"]
            novel_name = sample["novel_name"]
            row_idx = sample["row_idx"]
            text = sample["text"]

            text_ += text
            if len(text_) < args.min_text_length:
                continue

            row = {
                "text": text_
            }
            row = json.dumps(row, ensure_ascii=False)

            if random.random() < 0.95:
                ftrain.write("{}\n".format(row))
            else:
                fvalid.write("{}\n".format(row))

            text_ = ""
    return


if __name__ == '__main__':
    main()
