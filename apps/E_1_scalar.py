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


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

def generate_plots(all_as, all_bs, data_x, data_y, a, b_star):
    '''
    generates plots of convergence of a and b with different initialisations and also the final curve generated
    '''
    
    xs = np.linspace(data_x.numpy().min(), data_x.numpy().max())
    fig, ax = plt.subplots()
    ax.scatter(data_x.numpy(), data_y.numpy(), label="Data")
    ax.plot(xs, a.numpy()*xs**2 + (b_star.reshape((1,))).numpy(), color='red', lw=3, label="Learned Function")
    ax.legend(loc='best', fontsize=14)
    ax.set_title("Curve Fitting", fontsize=16)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    plt.savefig("results/scalar_example.png")
   

    fig, ax = plt.subplots()
    ax.plot(all_as[0], lw=3)
    ax.plot(all_as[1], lw=3)
    ax.plot(all_as[2], lw=3)
    ax.plot(all_as[3], lw=3)
    ax.axhline(1., color='#888888', lw=3, linestyle='--', label="Ground Truth")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(r"$a$ Value", fontsize=14)
    ax.set_title(r"Convergence of $a$", fontsize=16)
    ax.legend(loc='best', fontsize=14)
    plt.savefig("results/a_convergence.png")


    fig, ax = plt.subplots()
    ax.plot(all_bs[0], lw=3)
    ax.plot(all_bs[1], lw=3)
    ax.plot(all_bs[2], lw=3)
    ax.plot(all_bs[3], lw=3)
    ax.axhline(0.5, color='#888888', lw=3, linestyle='--', label="Ground Truth")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(r"$b$ Value", fontsize=14)
    ax.set_title(r"Convergence of $b$", fontsize=16)
    ax.legend(loc='best', fontsize=14)
    plt.savefig("results/b_convergence.png")
    plt.show()

def generate_data(num_points=100, a=1, b=0.5, noise_factor=0.05):
    '''
    Generate num_points of data sampled from a quadratic curve parameterized by a,b and noise factor
    '''
    
    data_x = init.rand(1, num_points, device=device)
    noise = init.randn(1, num_points, device=device) * noise_factor
    data_y = a * data_x**2 + b + noise

    x = Tensor(data_x, device=ndl.cpu())
    y = Tensor(data_y, device=ndl.cpu())
    return data_x, data_y, x, y 

def error_function(a, b, x, y):
    '''
    An example of a scalar error function defined over model outputs using needle ops
    '''
    xsquare = ops.power_scalar(x, 2)
    a_bd = ops.broadcast_to(a.reshape((1,1)), xsquare.shape)
    b_bd_1 = ops.broadcast_to(b.reshape((1,1)), xsquare.shape)
    ret = ops.power_scalar(xsquare*a_bd + b_bd_1 - y, 2)
    return ret

def run(model_optimizer, 
        num_epochs, 
        aux_vars, 
        optim_vars,
        implicit_layer):
    '''
    runs the optimization loop for number of epochs while using implicit layers
    '''
    # creating storing lists for a and b's intermediate values
    a_vals, b_vals = [aux_vars[0].numpy()[0]], [optim_vars.numpy()[0]]
    for epoch in tqdm(range(num_epochs)):
        model_optimizer.reset_grad()
        a, x, y = aux_vars
        
        # get optimum value by solving the inner optimization problem
        b_star = implicit_layer(a)
        
        # compute loss for outer optimization
        loss = error_function(a, b_star, x, y) 
        numel = loss.shape[1]
        loss = ops.summation(loss)
        loss = ops.divide_scalar(loss, numel)
        
        # compute gradients
        loss.backward()
        # modify params
        model_optimizer.step()

        # store the intermediate values
        a_vals.append(a.numpy()[0])
        b_vals.append(b_star.numpy()[0][0])

    return a, b_star, a_vals, b_vals

if __name__=='__main__':
    # generate sample data
    data_x, data_y, x, y  = generate_data()
    
    all_as, all_bs = [], []
    for a_val, b_val in zip([-2, 0, 2, 5], [-2, 0, 2, 5]):
        a = Tensor(init.ones(*(1,), requires_grad=True, device=ndl.cpu(),
                   dtype="float32")) * a_val
        b = Tensor(init.ones(*(1,), requires_grad=False, device=ndl.cpu(),
                   dtype="float32")) * b_val
        # set the variables to be optimized in the outer loop (aux vars)
        aux_vars = a, x, y

        # set the variables to be optimised in the inner loop (optim vars)
        optim_vars = b

        # set the outer loop optimizer with the appropiate params
        model_optimizer = ndl.optim.Adam([a], lr=1e-1, weight_decay=1e-3)

        # set the inner optimizer problem type (linear with scalar here)
        opt = "Scalar" 

        # define the cost function using one of our predefined cost functions
        cost_fn = ndl.implicit_cost_function.LinearCostFunction(aux_vars, 
                                                                optim_vars, 
                                                                error_function)
        
        # Initialize the implicit layer with the inner optim type, the cost function and the inner gradient calculation type
        implicit_layer = ndl.nn.ImplicitLayer(opt, cost_fn, "implicit")

        #num_epochs = 100
        num_epochs = 100

        # run the optimization for a given number of epochs 
        a, b_star, a_val, b_val = run(model_optimizer, num_epochs, aux_vars, optim_vars, implicit_layer)

        # error of final values of a and b compared with the ground truth values used for data generation (a=1, b=0.5)
        print("A ERROR: {}".format(np.linalg.norm(a.numpy() - 1)))
        print("B ERROR: {}".format(np.linalg.norm(b_star.numpy() - 0.5)))
        
        # tracking values of a and b as they evolved in optimization for plotting
        all_as.append(a_val)
        all_bs.append(b_val)

    generate_plots(all_as, all_bs,data_x, data_y, a, b_star )

