#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch


with open(r"D:\Users\tianx\PycharmProjects\PyTorch\trained_models\chinese_gpt2_h_novel\pytorch_model.bin", "rb") as f:
    state_dict = torch.load(f, map_location="cpu")

for k, v in state_dict.items():
    print(k)
# exit(0)
print(state_dict["transformer.h.1.mlp.c_fc.weight"][0:5])


with open(r"D:\Users\tianx\PycharmProjects\PyTorch\pretrained_models\gpt2-chinese-cluecorpussmall\pytorch_model.bin", "rb") as f:
    state_dict = torch.load(f, map_location="cpu")

# for k, v in state_dict.items():
#     print(k)
print(state_dict["transformer.h.1.mlp.c_fc.weight"][0:5])


if __name__ == '__main__':
    pass
