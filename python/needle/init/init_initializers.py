import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    val = 6.0/(fan_in+fan_out)
    a = gain*math.sqrt(val)
    c = rand(fan_in, fan_out, low =-a, high= a, **kwargs)
    return c
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    val = 2.0/(fan_in+fan_out)
    a = gain*math.sqrt(val)
    c = randn(fan_in, fan_out, mean =0, std= a, **kwargs)
    return c
    ### END YOUR SOLUTION



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    val = 3.0/fan_in
    gain = math.sqrt(2)
    a = gain*math.sqrt(val)
    if shape is None:
        c = rand(fan_in, fan_out, low =-a, high= a, **kwargs)
    else:
        c = rand(*shape, low =-a, high= a, **kwargs)
    return c
    ### END YOUR SOLUTION

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    val = 1.0/fan_in
    gain = math.sqrt(2)
    a = gain*math.sqrt(val)
    c = randn(fan_in, fan_out, mean=0, std= a, **kwargs)
    ### END YOUR SOLUTION