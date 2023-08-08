#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://zhuanlan.zhihu.com/p/486914187

序列化后的模型与 python 无关, 可以在 c++ 中运行.

序列化后的模型一般保存为 .pt 或 .pth

"""
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # 参数使用设置: 不要用 (3,) 形式, 用 3 或者 (3, 3).
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


def demo1():
    """pytorch模型序列化"""
    model = Model()
    model.eval()
    print(model)

    # trace 方式. 将模型运行一遍, 以记录对张量的操作并生成图模型.
    example_inputs = torch.randn((1, 3, 12, 12))
    model = torch.jit.trace(func=model, example_inputs=example_inputs)
    # script 方式. script 方式通过解析代码来生成图模型, 相较于 trace 方式, 它可以处理 if 条件判断的情况.
    # model = torch.jit.script(obj=model)

    # 模型序列化
    model.save('model.pth')
    return


def demo2():
    """加载 .pth 模型"""
    model = torch.jit.load('model.pth')
    print(model)

    inputs = torch.randn((1, 3, 12, 12))

    outputs = model.forward(inputs)
    print(outputs.shape)
    return


def demo3():
    """查看 .pth 模型"""
    model = torch.jit.load('model.pth')
    # 查看模型结构
    print(model)

    # 查看模型中的某一层
    model_conv1: torch.jit._script.RecursiveScriptModule = model.conv1

    # 查看
    print(model_conv1.graph)

    # 查看
    print(model_conv1.code)

    return


if __name__ == '__main__':
    demo1()
    # demo2()
    # demo3()
