"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in =self.kernel_size**2 *in_channels, fan_out = out_channels, shape=(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)), device=device)
        limit = 6*in_channels * kernel_size**2
        if bias:
            bias_init = init.kaiming_uniform(fan_in =limit, fan_out =1, shape=(self.out_channels,), device=device)
            self.bias = Parameter(bias_init)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        print('here')
        y = ops.transpose(x, (1, 2))
        z = ops.transpose(y, (2, 3))
        # print(z.shape)
        # print(x.shape)
        # compute padding here
        print(self.weight.device)
        padding = (self.kernel_size-1) // 2
        out_1 = ops.conv(z, self.weight, padding=padding, stride=self.stride)
        if self.bias is not None:
            new_b = [1]*len(out_1.shape)
            new_b[len(new_b)-1] = self.bias.shape[0]
            bias_2 = ops.broadcast_to(ops.reshape(self.bias, tuple(new_b)), out_1.shape)
            out_f = out_1 + bias_2
        else:
            out_f = out_1
        y_out = ops.transpose(out_f, (2, 3))
        z_out = ops.transpose(y_out, (1, 2))
        return z_out
        # first compute padding
        # then do conv op
        # then do broadcasted bias term
        ### END YOUR SOLUTION