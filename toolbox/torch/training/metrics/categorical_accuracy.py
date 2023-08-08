#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Iterable, List, Optional

import torch


class CategoricalAccuracy(object):

    supports_distributed = True

    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise AssertionError(
                "Tie break in Categorical Accuracy can be done only for maximum (top_k = 1)"
            )
        if top_k <= 0:
            raise AssertionError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        num_classes: torch.Tensor = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise AssertionError(
                "gold_labels must have dimension == predictions.size() - 1 but "
                "found tensor of shape: {}".format(predictions.size())
            )
        if (gold_labels >= num_classes).any():
            raise AssertionError(
                "A gold label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()

        if not self._tie_break:
            if self._top_k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1)
            else:
                _, sorted_indices = predictions.sort(dim=-1, descending=True)
                top_k = sorted_indices[..., : min(self._top_k, predictions.shape[-1])]

            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            correct = max_predictions_mask[
                torch.arange(gold_labels.numel(), device=gold_labels.device).long(), gold_labels
            ].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1)
            _total_count = mask.sum()
        else:
            _total_count = torch.tensor(gold_labels.numel())
        _correct_count = correct.sum()

        self.correct_count += _correct_count.item()
        self.total_count += _total_count.item()

    def get_metric(self, reset: bool = False) -> float:
        """
        # Returns

        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0

        if reset:
            self.reset()

        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
