#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch.nn as nn


name2activation = {
    'linear': lambda: lambda x: x,
    'relu': nn.ReLU,
}


if __name__ == '__main__':
    pass
