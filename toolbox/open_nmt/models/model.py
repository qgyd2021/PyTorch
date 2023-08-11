#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as f

from toolbox.open_nmt.common.registrable import Registrable


class Model(nn.Module, Registrable):
    pass
