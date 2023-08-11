#!/usr/bin/python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset

from toolbox.open_nmt.common.registrable import Registrable


class DatasetReader(Dataset, Registrable):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return NotImplementedError

    def read(self, filename: str) -> "DatasetReader":
        raise NotImplementedError


@DatasetReader.register("pair_dataset")
class PairDataset(DatasetReader):
    def __init__(self, sep: str = "\t"):
        self.sep = sep

        self.samples = list()

    def read(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            for row in f:
                row = str(row).strip().split(sep=self.sep)
                self.samples.append({
                    "src": row[0],
                    "tgt": row[1],
                })
        return len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]
