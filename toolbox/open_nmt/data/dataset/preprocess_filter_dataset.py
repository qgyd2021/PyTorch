#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List

from toolbox.open_nmt.data.dataset.dataset import DatasetReader
from toolbox.open_nmt.data.preprocess.preprocess import Preprocess
from toolbox.open_nmt.data.sample_filter.sample_filter import SampleFilter


@DatasetReader.register("preprocess_filter_dataset")
class PreprocessFilterDataset(DatasetReader):
    def __init__(self,
                 dataset: DatasetReader,
                 preprocess_list: List[Preprocess],
                 sample_filter_list: List[SampleFilter],
                 ):
        self.dataset = dataset
        self.preprocess_list = preprocess_list
        self.sample_filter_list = sample_filter_list

        self.samples = list()

        self._init()

    def _init(self):
        for sample in self.dataset:
            for preprocess in self.preprocess_list:
                sample = preprocess.process(sample)

            for sample_filter in self.sample_filter_list:
                flag: bool = sample_filter.filter(sample)
                if flag:
                    break
            else:
                self.samples.append(sample)

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)

    def read(self, filename: str) -> "DatasetReader":
        raise NotImplementedError


if __name__ == '__main__':
    pass
