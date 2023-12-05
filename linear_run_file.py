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
def generate_data(num_points=100, x=[[1., 1.]], y=0.1, noise_factor=0.001):
    if(not isinstance(x, np.ndarray)):
        x = Tensor(ndl.NDArray(np.array(x)), device=device)
    #if(not isinstance(y, np.ndarray)):
    #    y = Tensor(ndl.NDArray(np.array(y)), device=device)
    # Generate data: 100 points sampled from the quadratic curve listed above
    A = init.rand(2, num_points, device=device)
    #B = init.rand(2, num_points, device=device)
    noise = init.randn(1, num_points, device=device) * noise_factor
    #data_y = a * data_x**2 + b + noise

    #C = x @ data_x**2 + y + noise
    print(A.shape, x.shape)
    print(noise.shape)
    #print((x @ A + y @ B).shape)
    #C = x @ A + y @ B + noise
    B = x @ A + y + noise


    #A = Tensor(data_x, device=ndl.cpu())
    #B = Tensor(data_y, device=ndl.cpu())
    #return data_x, data_y, x, y 
    #return A, B, C, x, y
    return x, y, A, B

#def error_function(a, b, x, y):
def error_function(y, x, A, B):
    #xsquare = ops.power_scalar(x, 2)
    #a_bd = ops.broadcast_to(a.reshape((1,1)), xsquare.shape)
    #b_bd_1 = ops.broadcast_to(b.reshape((1,1)), xsquare.shape)
    #ret = ops.power_scalar(xsquare*a_bd + b_bd_1 - y, 2)

    #xsquare = ops.power_scalar(x, 2)
    #x_bd = ops.broadcast_to(x, A.shape)
    x_bd = x.reshape((1,2))
    print(x)
    print(y)
    y_bd = ops.broadcast_to(y.reshape((1,1)), (1, A.shape[1]))
    print(x_bd.shape,A.shape)
    print(y_bd.shape)
    print(B.shape)
    ret = ops.power_scalar(x_bd @ A + y_bd - B, 2)
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

        y, A, B = aux_vars
        x = optim_vars
        x_star = implicit_layer(y)
        print(x_star)
        #x_star = Parameter(Tensor([1.,1.], device=x.device))
        #b_star = b
        loss = error_function(y, x_star, A, B) #.mean()
        numel = loss.shape[1]
        loss = ops.summation(loss)
        loss = ops.divide_scalar(loss, numel)
        loss.backward()
        model_optimizer.step()
    print("\nFinal y and x")
    print(y, x_star)

if __name__=='__main__':
    data_x, data_y, A, B  = generate_data()
    # Plot the data
    #fig, ax = plt.subplots()
    #ax.scatter(data_x.numpy(), data_y.numpy())
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')

    ## plot fig ## 
    #plt.show()
    
    x = Tensor(init.ones(*(2,), requires_grad=True, device=ndl.cpu(), dtype="float32")) * 5.0
    y = Tensor(init.ones(*(1,), requires_grad=True, device=ndl.cpu(), dtype="float32")) * 5.0
    aux_vars = y, A, B
    optim_vars = x
    #raise

    model_optimizer = ndl.optim.Adam([y], lr=1e-1, weight_decay=1e-3)

    #opt = ndl.optim.InnerOptimizer(device='cpu')
    opt = "Linear" # or Nonlinear
    #opt = "None"
    cost_fn = ndl.implicit_cost_function.LinearCostFunction(aux_vars, 
                                                            optim_vars, 
                                                            error_function)
    implicit_layer = ndl.nn.ImplicitLayer(opt, cost_fn, "implicit")

    num_epochs = 1000

    run(model_optimizer, num_epochs, aux_vars, optim_vars, opt, cost_fn, implicit_layer)
    print("TARGET x and y")
    print(data_y, data_x)
    #print("\nHEY LOOK MA WE MADE IT\n")
    print("\nPEE PEE POO POO MA WE MADE IT\n")
    


