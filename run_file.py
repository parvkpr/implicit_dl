import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import matplotlib.pyplot as plt
from tqdm import tqdm

from needle.autograd import Tensor

import needle.init as init

import needle as ndl
from needle import backend_ndarray as nd
from needle import ops
from needle import backend_numpy

device = ndl.cpu()

# np.random.seed(1)

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

### STEP 0 -- generate data ###
def generate_data(num_points=100, a=1, b=0.5, noise_factor=0.001):
    # Generate data: 100 points sampled from the quadratic curve listed above
    data_x = init.rand(1, num_points, device=device)
    noise = init.randn(1, num_points, device=device) * noise_factor
    data_y = a * data_x**2 + b + noise

    x = Tensor(data_x, device=ndl.cpu())
    y = Tensor(data_y, device=ndl.cpu())
    return data_x, data_y, x, y 

def error_function(a, b, x, y):
    xsquare = ops.power_scalar(x, 2)
    #print(x.shape, xsquare.shape, y.shape)
    a_bd = ops.broadcast_to(a.reshape((1,1)), xsquare.shape)
    b_bd_1 = ops.broadcast_to(b.reshape((1,1)), xsquare.shape)
    ret = ops.power_scalar(xsquare*a_bd + b_bd_1 - y, 2)
    #raise
    return ret

def run(model_optimizer, 
        num_epochs, 
        aux_vars, 
        optim_vars,
        opt,
        cost_fn,
        implicit_layer):

    for epoch in tqdm(range(num_epochs)):
        model_optimizer.reset_grad()
        a, x, y = aux_vars
        b = optim_vars
        b_star = implicit_layer(a)
        #b_star = b
        loss = error_function(a, b_star, x, y) #.mean()
        numel = loss.shape[1]
        loss = ops.summation(loss)
        loss = ops.divide_scalar(loss, numel)
        loss.backward()
        model_optimizer.step()
    print("Final a and b")
    print(a, b_star)

if __name__=='__main__':
    data_x, data_y, x, y  = generate_data()
    # Plot the data
    fig, ax = plt.subplots()
    ax.scatter(data_x.numpy(), data_y.numpy())
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ## plot fig ## 
    #plt.show()
    
    a = Tensor(init.ones(*(1,), requires_grad=True, device=ndl.cpu(), dtype="float32")) * 5.0
    b = Tensor(init.ones(*(1,), requires_grad=False, device=ndl.cpu(), dtype="float32")) * 5.0
    aux_vars = a, x, y
    optim_vars = b
    #raise

    model_optimizer = ndl.optim.Adam([a], lr=1e-1, weight_decay=1e-3)

    #opt = ndl.optim.InnerOptimizer(device='cpu')
    #opt = "Linear" # or Nonlinear
    opt = "Nonlinear" # or Nonlinear
    #opt = "None"
    cost_fn = ndl.implicit_cost_function.LinearCostFunction(aux_vars, 
                                                            optim_vars, 
                                                            error_function)
    implicit_layer = ndl.nn.ImplicitLayer(opt, cost_fn, "implicit")

    num_epochs = 1000

    run(model_optimizer, num_epochs, aux_vars, optim_vars, opt, cost_fn, implicit_layer)
    print("\nHEY LOOK MA WE MADE IT\n")
    


