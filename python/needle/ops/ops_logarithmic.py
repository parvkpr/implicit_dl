from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_vals = Z.max(axis=self.axes, keepdims=True)
        r = Z - max_vals.broadcast_to(Z.shape)
        temp_r = array_api.exp(r)
        temp_r_s = array_api.log(temp_r.sum(axis=self.axes, keepdims=True))
        result = max_vals +temp_r_s
        #result = max_vals + array_api.log(array_api.sum(array_api.exp(Z - max_vals), axis=self.axes, keepdims=True))
        return result.sum(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        if isinstance(self.axes,int):
            self.axes = (self.axes, )
        denom = LogSumExp(self.axes)(x)
        new_axes = list(denom.shape)
        if(self.axes is not None):
          for a in self.axes:
            new_axes.insert(a, 1)

        denom = denom.reshape((new_axes)).broadcast_to(x.shape)
        out_grad = out_grad.reshape((new_axes)).broadcast_to(x.shape)
        temp_1 = x-denom
        expx = exp(temp_1)
        b = multiply(expx, out_grad)
        return b
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

