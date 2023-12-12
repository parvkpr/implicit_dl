"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class ImplicitLayer(Module):
    """
    this is our non linear least squares implicit layers
    """
    def __init__(self, inner_optimizer, cost_fn, implicit_grad_method, device=None, dtype="float32"):
        super().__init__()
        self.inner_optimizer = inner_optimizer
        self.cost_fn = cost_fn
        self.implicit_grad_method = implicit_grad_method
        self.device = device
        self.dtype = dtype


    def forward(self, x:Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        _, A, B = self.cost_fn.aux_vars
        y = self.cost_fn.optim_vars
        Z = ops.lsimplicit(self.inner_optimizer, self.implicit_grad_method, x, y, A, B)
        return Z
        ### END YOUR SOLUTION


class WeightImplicitLayer(Module):
    """
    this is our non linear least squares implicit layers
    """
    def __init__(self, inner_optimizer, cost_fn, implicit_grad_method, device=None, dtype="float32"):
        super().__init__()
        self.inner_optimizer = inner_optimizer
        self.cost_fn = cost_fn
        self.implicit_grad_method = implicit_grad_method
        self.device = device
        self.dtype = dtype


    def forward(self, w1, w2, x:Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Z = ops.weight_lsimplicit(self.inner_optimizer, self.cost_fn,
                                  self.implicit_grad_method, x, w1, w2)
        return Z
        ### END YOUR SOLUTION
