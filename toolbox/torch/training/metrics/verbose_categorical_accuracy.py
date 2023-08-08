#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, List, Optional

import numpy as np
import torch


class CategoricalAccuracyVerbose(object):
    def __init__(self,
                 index_to_token: Dict[int, str],
                 label_namespace: str = "labels",
                 top_k: int = 1,
                 ) -> None:
        if top_k <= 0:
            raise AssertionError("top_k passed to Categorical Accuracy must be > 0")
        self._index_to_token = index_to_token
        self._label_namespace = label_namespace
        self._top_k = top_k
        self.correct_count = 0.
        self.total_count = 0.
        self.label_correct_count = dict()
        self.label_total_count = dict()

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise AssertionError("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions.size()))
        if (gold_labels >= num_classes).any():
            raise AssertionError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()

        # Top K indexes of the predictions (or fewer, if there aren't K of them).
        # Special case topk == 1, because it's common and .max() is much faster than .topk().
        if self._top_k == 1:
            top_k = predictions.max(-1)[1].unsqueeze(-1)
        else:
            top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

        # This is of shape (batch_size, ..., top_k).
        correct = top_k.eq(gold_labels.unsqueeze(-1)).float()

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

        labels: List[int] = np.unique(gold_labels.cpu().numpy()).tolist()
        for label in labels:
            label_mask = (gold_labels == label)

            label_correct = correct * label_mask.view(-1, 1).float()
            label_correct = int(label_correct.sum())
            label_count = int(label_mask.sum())

            label_str = self._index_to_token[label]
            if label_str in self.label_correct_count:
                self.label_correct_count[label_str] += label_correct
            else:
                self.label_correct_count[label_str] = label_correct

            if label_str in self.label_total_count:
                self.label_total_count[label_str] += label_count
            else:
                self.label_total_count[label_str] = label_count

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        result = dict()
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        result['accuracy'] = accuracy

        for label in self.label_total_count.keys():
            total = self.label_total_count[label]
            correct = self.label_correct_count.get(label, 0.0)
            label_accuracy = correct / total
            result[label] = label_accuracy

        if reset:
            self.reset()
        return result

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
        self.label_correct_count = dict()
        self.label_total_count = dict()


def demo1():

    categorical_accuracy_verbose = CategoricalAccuracyVerbose(
        index_to_token={0: '0', 1: '1'},
        top_k=2,
    )

    predictions = torch.randn(size=(2, 3), dtype=torch.float32)
    gold_labels = torch.ones(size=(2,), dtype=torch.long)
    # print(predictions)
    # print(gold_labels)

    categorical_accuracy_verbose(
        predictions=predictions,
        gold_labels=gold_labels,
    )
    metric = categorical_accuracy_verbose.get_metric()
    print(metric)
    return


if __name__ == '__main__':
    demo1()
