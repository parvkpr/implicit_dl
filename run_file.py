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
def generate_data(num_points=100, a=1, b=0.5, noise_factor=0.05):
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

    a_vals, b_vals = [aux_vars[0].numpy()[0]], [optim_vars.numpy()[0]]
    for epoch in tqdm(range(num_epochs)):
        model_optimizer.reset_grad()
        a, x, y = aux_vars
        #b = optim_vars
        b_star = implicit_layer(a)
        #b_star = b
        loss = error_function(a, b_star, x, y) #.mean()
        numel = loss.shape[1]
        loss = ops.summation(loss)
        loss = ops.divide_scalar(loss, numel)
        loss.backward()
        model_optimizer.step()
        a_vals.append(a.numpy()[0])
        b_vals.append(b_star.numpy()[0])

    return a, b_star, a_vals, b_vals

if __name__=='__main__':
    data_x, data_y, x, y  = generate_data()
    # Plot the data
    
    all_as, all_bs = [], []
    for a_val, b_val in zip([-2, 0, 2, 5], [-2, 0, 2, 5]):
        a = Tensor(init.ones(*(1,), requires_grad=True, device=ndl.cpu(),
                   dtype="float32")) * a_val
        b = Tensor(init.ones(*(1,), requires_grad=False, device=ndl.cpu(),
                   dtype="float32")) * b_val
        aux_vars = a, x, y
        optim_vars = b

        model_optimizer = ndl.optim.Adam([a], lr=1e-1, weight_decay=1e-3)

        opt = "Scalar" # or Nonlinear
        #opt = "Linear" # or Nonlinear
        cost_fn = ndl.implicit_cost_function.LinearCostFunction(aux_vars, 
                                                                optim_vars, 
                                                                error_function)
        implicit_layer = ndl.nn.ImplicitLayer(opt, cost_fn, "implicit")

        num_epochs = 100

        a, b_star, a_val, b_val = run(model_optimizer, num_epochs, aux_vars, optim_vars, opt, cost_fn, implicit_layer)
        print("\nHEY LOOK MA WE MADE IT\n")
        #print("TARGET A: {} GOT A: {}".format(a_val, a))
        #print("TARGET b: {} GOT b: {}".format(b_val, b_star))
        print("A ERROR: {}".format(np.linalg.norm(a.numpy() - 1)))
        print("B ERROR: {}".format(np.linalg.norm(b_star.numpy() - 0.5)))
        all_as.append(a_val)
        all_bs.append(b_val)
        #raise

    #xs = np.linspace(data_x.numpy().min(), data_x.numpy().max())
    #fig, ax = plt.subplots()
    #ax.scatter(data_x.numpy(), data_y.numpy(), label="Data")
    #ax.plot(xs, a.numpy()*xs**2 + b_star.numpy(), color='red', lw=3, label="Learned Function")
    #ax.legend(loc='best', fontsize=14)
    #ax.set_title("Curve Fitting", fontsize=16)
    #ax.set_xlabel('x', fontsize=16)
    #ax.set_ylabel('y', fontsize=16)
    #plt.savefig("scalar_example.png")
    #plt.savefig("scalar_example.pdf")

    ## plot fig ## 
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
    plt.savefig("a_convergence.png")
    plt.savefig("a_convergence.pdf")

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
    plt.savefig("b_convergence.png")
    plt.savefig("b_convergence.pdf")
    plt.show()

