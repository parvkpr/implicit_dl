import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import matplotlib.pyplot as plt
from needle.autograd import Tensor
import needle.init as init

import needle as ndl
from needle import backend_ndarray as nd
from needle import ops
from needle import backend_numpy

# np.random.seed(1)

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

### STEP 0 -- generate data ###
def generate_data(num_points=100, a=1, b=0.5, noise_factor=0.01):
    # Generate data: 100 points sampled from the quadratic curve listed above
    data_x = np.random.rand(1, num_points)
    noise = np.random.randn(1, num_points) * noise_factor
    data_y = a * data_x**2 + b + noise

    x = Tensor(data_x)
    y = Tensor(data_y)
    return data_x, data_y, x, y 

def error_function(a, b, x, y):
    xsquare = ops.power_scalar(x, 2)
    a_bd = ops.broadcast_to(a, xsquare.shape)
    b_bd = ops.broadcast_to(b, xsquare.shape)
    ret = ops.multiply(xsquare, a_bd) + b_bd - y

def run(model_optimizer, 
        num_epochs, 
        aux_vars, 
        optim_vars,
        opt,
        cost_fn,
        implicit_layer):

    for epoch in range(num_epochs):
        model_optimizer.reset_grad()
        a, x, y = aux_vars
        b = optim_vars
        b_star = implicit_layer(a)
        loss = error_function(a, b_star, x, y).mean()
        loss.backward()
        model_optimizer.step()


if __name__=='__main__':
    data_x, data_y, x, y  = generate_data()
    # Plot the data
    fig, ax = plt.subplots()
    ax.scatter(data_x, data_y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ## plot fig ## 
    #plt.show()
    
    a = Parameter(init.ones(*(1,), requires_grad=True, device=None, dtype="float32"))
    b = Parameter(init.ones(*(1,), requires_grad=False, device=None, dtype="float32"))
    aux_vars = a, x, y
    optim_vars = b

    model_optimizer = ndl.optim.Adam([a], lr=1e-3, weight_decay=1e-3)

    opt = ndl.optim.InnerOptimizer(device='cpu')
    cost_fn = ndl.implicit_cost_function.LinearCostFunction(aux_vars, 
                                                            optim_vars, 
                                                            error_function)
    implicit_layer = ndl.nn.ImplicitLayer(opt, cost_fn, "implicit")

    num_epochs = 20

    run(model_optimizer, num_epochs, aux_vars, optim_vars, opt, cost_fn, implicit_layer)
    


