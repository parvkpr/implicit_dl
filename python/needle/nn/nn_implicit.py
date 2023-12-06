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
        Z = ops.lsimplicit(self.inner_optimizer, self.cost_fn, self.implicit_grad_method, x)
        #Z = ops.tanh(x)
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



    def forward(self, w1:Tensor, w2:Tensor, x:Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Set weights in each cost function
        #print("\nWEIGHT IMPLICIT\n")
        #self.cost_fn[0].w = w1
        #self.cost_fn[1].w = w2
        #Z = ops.lsimplicit(self.inner_optimizer, self.cost_fn, self.implicit_grad_method, x)
        Z = ops.weight_lsimplicit(self.inner_optimizer, self.cost_fn,
                                  self.implicit_grad_method, w1, w2, x)
        #Z = ops.tanh(x)
        return Z
        ### END YOUR SOLUTION
