#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.bert.tokenization_bert import BertTokenizer

from project_settings import project_path


def get_args():
    """
python3 3.test_model.py \
--repetition_penalty 1.2 \
--trained_model_path /data/tianxing/PycharmProjects/Transformers/trained_models/gpt2_chinese_h_novel

python3 3.test_model.py \
--trained_model_path /data/tianxing/PycharmProjects/Transformers/trained_models/gpt2_chinese_h_novel

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trained_model_path',
        default=(project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix(),
        type=str,
    )
    parser.add_argument('--device', default='auto', type=str)

    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--temperature', default=0.35, type=float)
    parser.add_argument('--repetition_penalty', default=1.2, type=float)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # pretrained model
    tokenizer = BertTokenizer.from_pretrained(args.trained_model_path)
    model = GPT2LMHeadModel.from_pretrained(args.trained_model_path)

    model.eval()
    model = model.to(device)

    while True:
        text = input('prefix: ')

        if text == "Quit":
            break
        text = '{}'.format(text)
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids[:, :-1]
        # print(input_ids)
        # print(type(input_ids))
        input_ids = input_ids.to(device)

        outputs = model.generate(input_ids,
                                 max_new_tokens=512,
                                 do_sample=True,
                                 top_p=args.top_p,
                                 temperature=args.temperature,
                                 repetition_penalty=args.repetition_penalty,
                                 # eos_token_id=tokenizer.sep_token_id,
                                 eos_token_id=None,
                                 pad_token_id=tokenizer.pad_token_id
                                 )
        rets = tokenizer.batch_decode(outputs)
        output = rets[0].replace(" ", "").replace("[CLS]", "").replace("[SEP]", "")
        print("{}".format(output))

    return


if __name__ == '__main__':
    main()
