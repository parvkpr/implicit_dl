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
def generate_data(num_points=200, x=[[1.]*50], y=0.1, noise_factor=0.01):
    if(not isinstance(x, np.ndarray)):
        x = Tensor(ndl.NDArray(np.array(x)), device=device)

    # Generate data and noise
    A = init.rand(x.shape[1], num_points, device=device)
    noise = init.randn(1, num_points, device=device) * noise_factor

    # Linear model
    B = x @ A + y + noise

    return x, y, A, B

#def error_function(a, b, x, y):
def error_function(y, x, A, B):
    #xsquare = ops.power_scalar(x, 2)

    #xsquare = ops.power_scalar(x, 2)
    x_bd = x.reshape((1,x.shape[0]))
    y_bd = ops.broadcast_to(y.reshape((1,1)), (1, A.shape[1]))

    # Use known functional form
    ret = ops.power_scalar(x_bd @ A + y_bd - B, 2)
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

        y, A, B = aux_vars
        x = optim_vars
        x_star = implicit_layer(y)
        #x_star = x
        #(x_star**2).backward()
        #x_star.backward()
        #print(y)
        #print(y.grad)
        #raise
        #x_star = Parameter(Tensor([1.,1.], device=x.device))
        #b_star = b
        loss = error_function(y, x_star, A, B) #.mean()
        #loss = error_function(y, x, A, B) #.mean()
        numel = loss.shape[1]
        loss = ops.summation(loss)
        loss = ops.divide_scalar(loss, numel)
        #print(loss)
        loss.backward()
        #print(x.grad)
        #print(y.grad)
        model_optimizer.step()
    print("\nFinal y and x")
    print(y, x_star)
    return y, x_star

if __name__=='__main__':
    np.random.seed(137)
    NUM_X = 5
    input_x = list(np.random.randn(NUM_X)[None])
    data_x, data_y, A, B  = generate_data(x=input_x)
    #print(A)

    # Plot the data
    #fig, ax = plt.subplots()
    #ax.scatter(data_x.numpy(), data_y.numpy())
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #plt.show()
    
    x = Tensor(init.ones(*(data_x.shape[1],), requires_grad=True, device=ndl.cpu(), dtype="float32")) * 1.0
    y = Tensor(init.ones(*(1,), requires_grad=True, device=ndl.cpu(), dtype="float32")) * 1.0
    aux_vars = y, A, B
    optim_vars = x

    model_optimizer = ndl.optim.Adam([y], lr=1e-1, weight_decay=1e-3)

    #opt = ndl.optim.InnerOptimizer(device='cpu')
    #opt = "Linear" # or Nonlinear
    opt = "Linear"
    cost_fn = ndl.implicit_cost_function.LinearCostFunction(aux_vars, 
                                                            optim_vars, 
                                                            error_function)
    implicit_layer = ndl.nn.ImplicitLayer(opt, cost_fn, "implicit")

    num_epochs = 1000

    final_y, final_x = run(model_optimizer, num_epochs, aux_vars, optim_vars, opt, cost_fn, implicit_layer)
    print("TARGET x and y")
    print(data_y, data_x)
    print("\nHEY LOOK MA WE MADE IT\n")
    print("Y ERROR: {}".format(np.linalg.norm(data_y - final_y.numpy())))
    print("X ERROR: {}".format(np.linalg.norm(data_x.numpy() - final_x.numpy())))


