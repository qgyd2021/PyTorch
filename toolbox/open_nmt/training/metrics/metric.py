#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, cast, Iterable, Optional

import torch

from toolbox.open_nmt.common.registrable import Registrable
from toolbox.open_nmt.data.collate_functions.collate_fn import CollateFunction
from toolbox.open_nmt.models.model import Model
from toolbox.open_nmt.data.vocabulary import Vocabulary


class Metric(Registrable):

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]
    ):
        raise NotImplementedError

    def get_metric(self, reset: bool):
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)


if __name__ == '__main__':
    pass
