import sys
sys.path.append("../python/")
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

def generate_data(num_points=100, x=[[1.]*50], y=0.1, noise_factor=0.01):
    if(not isinstance(x, np.ndarray)):
        x = Tensor(ndl.NDArray(np.array(x)), device=device)

    # Generate data and noise
    A = init.rand(x.shape[1], num_points, device=device)
    noise = init.randn(1, num_points, device=device) * noise_factor

    # Linear model
    B = x @ A + y + noise

    return x, y, A, B


def error_function(y, x, A, B):
    '''
    An example of an error function defined over model outputs using needle ops
    '''
    x_bd = x.reshape((1,x.shape[0]))
    y_bd = ops.broadcast_to(y.reshape((1,1)), (1, A.shape[1]))
    ret = ops.power_scalar(x_bd @ A + y_bd - B, 2)
    return ret


def compute_norms(X, XGt, Y, YGt):
    '''
    compute norms of model outputs and ground truth data
    '''
    norm_X = np.linalg.norm(X.numpy() - XGt, axis=1)
    norm_Y = np.linalg.norm(Y - YGt)
    return norm_X, norm_Y

def generate_plots(data):
    '''
    generates plots of error norms of X and Y and their evolution with size
    '''
    nx = []
    ny = []
    sizes = []

    for X, XGt, Y, YGt in data:
        norm_x, norm_y = compute_norms(X, XGt, Y, YGt)
        nx.append(norm_x)
        ny.append(norm_y)
        sizes.append(X.shape[1])


    fig, ax = plt.subplots(figsize=(10,5), ncols=2)

    # Plot for X-XGt
    ax[0].plot(sizes, nx, label=f'Data', lw=3, markersize=10, marker='o')
    ax[0].set_xlabel('Size of X', fontsize=14)
    ax[0].set_ylabel('Norm', fontsize=14)
    ax[0].set_title('Norm of X - XGt', fontsize=16)
    ax[0].set_ylim(0, 0.08)
    ax[0].legend()

    # Plot for Y-YGt
    ax[1].plot(sizes, ny, label=f'Data', lw=3, markersize=10, marker='o')
    ax[1].set_xlabel('Size of X', fontsize=14)
    ax[1].set_title('Norm of Y - YGt', fontsize=16)
    ax[1].set_ylim(0, 0.04)
    ax[1].legend()

    # Show plots
    plt.savefig("results/scaling.png")
    plt.show()



def run(model_optimizer, 
        num_epochs, 
        aux_vars, 
        implicit_layer):
    '''
    runs the optimization loop for number of epochs while using implicit layers
    '''
    for epoch in tqdm(range(num_epochs)):
        model_optimizer.reset_grad()

        y, A, B = aux_vars
        # get optimum value by solving the inner optimization problem
        x_star = implicit_layer(y)
        
        # compute loss for outer optimization
        loss = error_function(y, x_star, A, B) 
        numel = loss.shape[1]
        loss = ops.summation(loss)
        loss = ops.divide_scalar(loss, numel)
        
        # compute gradients
        loss.backward()

        # modify params
        model_optimizer.step()

    return y, x_star

if __name__=='__main__':
    np.random.seed(137)
    data = []
    for NUM_X in [2, 5, 10, 20, 30]:
        # generate sample data for a given size of x
        input_x = list(np.random.randn(NUM_X)[None])
        data_x, data_y, A, B  = generate_data(x=input_x)
        
        
        x = Tensor(init.ones(*(data_x.shape[1],), requires_grad=True, device=ndl.cpu(), dtype="float32")) * 1.0
        y = Tensor(init.ones(*(1,), requires_grad=True, device=ndl.cpu(), dtype="float32")) * 1.0
        
        # set the variables to be optimised in the outer loop (aux vars)
        aux_vars = y, A, B
        # set the variables to be optimised in the inner loop (optim vars)
        optim_vars = x

        # set the outer loop optimizer with the appropiate params
        model_optimizer = ndl.optim.Adam([y], lr=1e-1, weight_decay=1e-7)

        # set the inner optimizer problem type (linear least squares here)
        opt = "Linear"

        # define the cost function using one of our predefined cost functions
        cost_fn = ndl.implicit_cost_function.LinearCostFunction(aux_vars, 
                                                                optim_vars, 
                                                                error_function)
        
        #Initialize the implicit layer with the inner optim type, the cost function and the inner gradient calculation type
        implicit_layer = ndl.nn.ImplicitLayer(opt, cost_fn, "implicit")


        num_epochs = 10000

        # run the optimization for a given number of epochs
        final_y, final_x = run(model_optimizer, num_epochs, aux_vars, implicit_layer)
 
        print("Y ERROR: {}".format(np.linalg.norm(data_y - final_y.numpy())))
        print("X ERROR: {}".format(np.linalg.norm(data_x.numpy() - final_x.numpy())))

        # store data for plotting
        data.append([data_x, final_x.numpy(), data_y, final_y.numpy()])

    # generate the plots from error of final predictions and size of x
    generate_plots(data)


