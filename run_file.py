import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import torch

import needle as ndl
from needle import backend_ndarray as nd

np.random.seed(1)


if __name__=='__main__':
    #def __init__(self, inner_optimizer, cost_fn, implicit_grad_method, device=None, dtype="float32"):
    opt = ndl.optim.InnerOptimizer(device='cpu')
    cost_fn = ndl.implicit_cost_function.CostFunction(None, None)
    a = ndl.nn.Implicit(opt,cost_fn, "implicit")
    c = a(ndl.Tensor(0))
    # print(c)
    # print(a.cost_fn)
    # print(a.inner_optimizer)
    # a
    # b = ndl.nn.LU()

