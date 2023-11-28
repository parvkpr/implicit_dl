"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in =in_features, fan_out = out_features), device=device, dtype=dtype)


        bias_init = init.kaiming_uniform(fan_in =out_features, fan_out =1)
        self.bias = Parameter(ops.transpose(bias_init), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out_1 = ops.matmul(X, self.weight)
        bias_2 = ops.broadcast_to(self.bias, out_1.shape)
        out_f = out_1 + bias_2
        return out_f
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        flattened_shape = (batch_size, -1)
        b = ops.reshape(X, flattened_shape) 
        return b
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        tot = []
        tot.append(x)
        for i, l in enumerate(self.modules):
          tot.append(l(tot[i]))
        return tot[-1]
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        v_3 = ops.logsumexp(logits, (1,))
        y_one_hot = init.one_hot(logits.shape[1], y)
        v_4 = ops.EWiseMul()(logits, y_one_hot)
        v_5 = ops.Summation((1,))(v_4)
        temp2 = ops.Negate()(v_5)
        temp_3 = ops.EWiseAdd()(v_3, temp2)

        temp_4 = ops.DivScalar(logits.shape[0])(temp_3)
        temp_5 = ops.Summation((0,))(temp_4)
        return temp_5
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training is True:
          temp_1 = ops.summation(x, axes=(0,))
          num = x.shape[0]
          mean = ops.divide_scalar(temp_1, num)
          self.running_mean = (1-self.momentum)*self.running_mean+ self.momentum*mean
          mean_new = ops.reshape(mean, (1,-1))
          mean_new_2 = ops.broadcast_to(mean_new, x.shape)
          temp_diff = x - mean_new_2
          temp_diff_1 = ops.power_scalar(temp_diff, 2)
          temp_diff_2 = ops.summation(temp_diff_1, axes=(0,))
          var = ops.divide_scalar(temp_diff_2, num)
          self.running_var = (1-self.momentum)*self.running_var+ self.momentum*var
          var_new = ops.reshape(var, (1, -1))
          var_new_2 = ops.broadcast_to(var_new, x.shape)
        else:
          var_new_2 =self.running_var
          mean_new_2 =self.running_mean
          mean_new_2 = ops.reshape(mean_new_2, (1,-1))
          mean_new_2 = ops.broadcast_to(mean_new_2, x.shape)
          var_new_2 = ops.reshape(var_new_2, (1, -1))
          var_new_2 = ops.broadcast_to(var_new_2, x.shape)

        
        var_eps = var_new_2 + self.eps
        var_eps_root = ops.power_scalar(var_eps, 0.5)
        inp = ops.divide(x-mean_new_2, var_eps_root)
        weight = self.weight
        bias = self.bias
        weight = ops.broadcast_to(ops.reshape(weight, (1,-1)), x.shape)
        bias = ops.broadcast_to(ops.reshape(bias, (1, -1)), x.shape)
        return weight*inp +bias
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        temp_1 = ops.summation(x, axes=(1,))
        num = x.shape[1]
        mean = ops.divide_scalar(temp_1, num)
        mean_new = ops.reshape(mean, (-1, 1))
        mean_new_2 = ops.broadcast_to(mean_new, x.shape)
        temp_diff = x - mean_new_2
        temp_diff_1 = ops.power_scalar(temp_diff, 2)
        temp_diff_2 = ops.summation(temp_diff_1, axes=(1,))
        var = ops.divide_scalar(temp_diff_2, num)

        var_new = ops.reshape(var, (-1, 1))
        var_new_2 = ops.broadcast_to(var_new, x.shape)

        
        var_eps = var_new_2 + self.eps
        var_eps_root = ops.power_scalar(var_eps, 0.5)
        inp = ops.divide(temp_diff, var_eps_root)
        weight = self.weight
        bias = self.bias
        weight = ops.broadcast_to(ops.reshape(weight, (1, weight.shape[0])), x.shape)
        bias = ops.broadcast_to(ops.reshape(bias, (1, bias.shape[0])), x.shape)
        return weight*inp +bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
          return x

        #dropout_mask = np.random.binomial(1, 1 - self.p, size=x.shape)
        dr = init.rand(*x.shape, low=0, high=1).numpy() < 1-self.p
        dr = Tensor(dr, dtype="float32")
        # Scale the output by 1 / (1 - dropout_prob) to maintain expected value
        output_tensor = ops.multiply(dr,x)/(1 - self.p)
        return output_tensor
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
