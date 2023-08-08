#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch


device = torch.device("cpu:0")
print(device)
device = torch.device("cpu:1")
print(device)


if __name__ == '__main__':
    pass
