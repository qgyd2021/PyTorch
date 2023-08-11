#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, cast, Dict, List

import torch

from toolbox.open_nmt.common.registrable import Registrable
from toolbox.open_nmt.data.collate_functions.collate_fn import CollateFunction
from toolbox.open_nmt.models.model import Model
from toolbox.open_nmt.data.vocabulary import Vocabulary


class Predictor(Registrable):
    def __init__(self,
                 model: Model,
                 collate_fn: CollateFunction,
                 device: torch.device = None,
                 ):
        self.model = model.to(device)
        self.collate_fn = collate_fn
        self.device = device or torch.device('cpu')

    def predict_json(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_batch_json(self, inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError


class BasicPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 collate_fn: CollateFunction,
                 vocabulary: Vocabulary,
                 device: torch.device = None,
                 ):
        super().__init__(model=model, collate_fn=collate_fn, device=device)
        self.vocabulary = vocabulary

    def predict_json(self,
                     inputs: Dict[str, Any],
                     mode: str = "greedy_decode",
                     mode_kwargs: dict = None
                     ) -> Dict[str, Any]:
        outputs = self.predict_batch_json(inputs=[inputs], mode=mode)
        return outputs[0]

    def predict_batch_json(self,
                           inputs: List[Dict[str, Any]],
                           mode: str = "greedy_decode",
                           mode_kwargs: dict = None
                           ) -> List[Dict[str, Any]]:
        mode_kwargs = mode_kwargs or dict()

        fn = getattr(self.model, mode)

        batch = self.collate_fn(inputs)
        batch = batch[:2]
        batch = [x.to(self.device) for x in batch]

        with torch.no_grad():
            hypotheses = fn(*batch, **mode_kwargs)

        # clear cuda cache
        torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        cast(List[List[int]], hypotheses)

        hypotheses = [
            " ".join([self.vocabulary.get_token_from_index(idx, namespace="tgt_tokens") for idx in seq])
            for seq in hypotheses
        ]

        outputs = [
            {
                'hypothesis': hypothesis,
            } for hypothesis in hypotheses
        ]
        return outputs
