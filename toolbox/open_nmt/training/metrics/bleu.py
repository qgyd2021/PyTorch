#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import Counter
import math
from typing import Iterable, Tuple, Dict, Set, Optional

import torch

from toolbox.open_nmt.training.metrics.metric import Metric
from toolbox.open_nmt.training.util import get_valid_tokens_mask, ngrams


@Metric.register("bleu")
class BLEU(Metric):
    def __init__(
        self,
        ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
        exclude_indices: Set[int] = None,
    ) -> None:
        self._ngram_weights = ngram_weights
        self._exclude_indices = exclude_indices or set()
        self._precision_matches: Dict[int, int] = Counter()
        self._precision_totals: Dict[int, int] = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    def _get_modified_precision_counts(
        self,
        predicted_tokens: torch.LongTensor,
        reference_tokens: torch.LongTensor,
        ngram_size: int,
    ) -> Tuple[int, int]:
        clipped_matches = 0
        total_predicted = 0

        for predicted_row, reference_row in zip(predicted_tokens, reference_tokens):
            predicted_ngram_counts = ngrams(predicted_row, ngram_size, self._exclude_indices)
            reference_ngram_counts = ngrams(reference_row, ngram_size, self._exclude_indices)
            for ngram, count in predicted_ngram_counts.items():
                clipped_matches += min(count, reference_ngram_counts[ngram])
                total_predicted += count
        return clipped_matches, total_predicted

    def _get_brevity_penalty(self) -> float:
        if self._prediction_lengths > self._reference_lengths:
            return 1.0
        if self._reference_lengths == 0 or self._prediction_lengths == 0:
            return 0.0
        return math.exp(1.0 - self._reference_lengths / self._prediction_lengths)

    def __call__(
        self,
        predictions: torch.LongTensor,
        gold_targets: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        """
        :param predictions: torch.LongTensor,
            Batched predicted tokens of shape=[batch_size, max_sequence_length].
        :param gold_targets: torch.LongTensor,
            Batched reference (gold) translations with shape=[batch_size, max_gold_sequence_length].
        :param mask:
        :return:
        """
        if mask is not None:
            raise NotImplementedError("This metric does not support a mask.")

        predictions, gold_targets = self.detach_tensors(predictions, gold_targets)

        for ngram_size, _ in enumerate(self._ngram_weights, start=1):
            precision_matches, precision_totals = self._get_modified_precision_counts(
                predictions, gold_targets, ngram_size
            )

            self._precision_matches[ngram_size] += precision_matches
            self._precision_totals[ngram_size] += precision_totals

        if not self._exclude_indices:
            _prediction_lengths = predictions.size(0) * predictions.size(1)
            _reference_lengths = gold_targets.size(0) * gold_targets.size(1)

        else:
            valid_predictions_mask = get_valid_tokens_mask(predictions, self._exclude_indices)
            valid_gold_targets_mask = get_valid_tokens_mask(gold_targets, self._exclude_indices)
            _prediction_lengths = valid_predictions_mask.sum().item()
            _reference_lengths = valid_gold_targets_mask.sum().item()

        self._prediction_lengths += _prediction_lengths
        self._reference_lengths += _reference_lengths

    def get_metric(self, reset: bool = False) -> Dict[str, float]:

        brevity_penalty = self._get_brevity_penalty()
        ngram_scores = (
            weight
            * (
                math.log(self._precision_matches[n] + 1e-13)
                - math.log(self._precision_totals[n] + 1e-13)
            )
            for n, weight in enumerate(self._ngram_weights, start=1)
        )
        bleu = brevity_penalty * math.exp(sum(ngram_scores))

        if reset:
            self.reset()
        return {"BLEU": bleu}

    def reset(self) -> None:
        self._precision_matches = Counter()
        self._precision_totals = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0


if __name__ == '__main__':
    pass
