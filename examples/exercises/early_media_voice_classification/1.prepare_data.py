#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import random

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=(project_path / "examples/exercises/early_media_voice_classification/data_dir/").as_posix(),
        type=str
    )
    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)
    parser.add_argument("--train_rate", default=0.9, type=float)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    data_path = Path(args.data_path)

    label_map = {
        "bell": "bell",
        "music": "music",
        "mute": "mute",
        "voice": "voice",
    }

    with open(args.train_subset, "w", encoding="utf-8") as ftrain, \
        open(args.valid_subset, "w", encoding="utf-8") as fvalid:
        for filename in data_path.glob("*/*_segmented/*/*.wav"):
            label = filename.parts[-2]

            if label not in label_map.keys():
                continue

            row = {
                "filename": filename.as_posix(),
                "label": label_map[label],
            }
            row = json.dumps(row, ensure_ascii=False)

            if random.random() < args.train_rate:
                ftrain.write("{}\n".format(row))
            else:
                fvalid.write("{}\n".format(row))

    return


if __name__ == '__main__':
    main()
