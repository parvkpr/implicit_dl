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


def generate_plots(bstars):
    '''
    generates plots of convergence of b with different initalizations
    '''
    fig, ax = plt.subplots()
    ax.plot(bstars[0], lw=3)
    ax.plot(bstars[1], lw=3)
    ax.plot(bstars[2], lw=3)
    ax.plot(bstars[3], lw=3)

    ax.axhline(-0.5, color='#888888', lw=3, linestyle='--', label="Ground Truth")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(r"$\frac{-w2}{w1+w2}$", fontsize=14)
    ax.set_title(r"Convergence of $b^*$", fontsize=16)
    ax.legend(loc='best', fontsize=14)
    plt.savefig("results/w_convergence.png")
    plt.savefig("results/w_convergence.pdf")
    plt.show()

def generate_data(num_points=1000, a=1, b=0.5, noise_factor=0.01):
    '''
    Generate num_points of data sampled from a nonlinear function (tanh) parameterized by a,b and noise factor
    '''
    data_x = init.rand(1, num_points, device=device)*2 - 1
    noise = init.randn(1, num_points, device=device) * noise_factor

    data_y = a * ops.tanh(data_x) + b + noise
    x = Tensor(data_x, device=ndl.cpu())
    y = Tensor(data_y, device=ndl.cpu())
    return data_x, data_y, x, y, Tensor(np.array([a]), device=ndl.cpu()), b

def error_function(a, b, x, y):
    '''
    An example of an error function defined over model outputs using needle ops
    '''
    xsquare = ops.tanh(x)
    a_bd = ops.broadcast_to(a.reshape((1,1)), xsquare.shape)
    b_bd_1 = ops.broadcast_to(b.reshape((1,1)), xsquare.shape)
    ret = ops.power_scalar(ops.EWiseMul()(xsquare, a_bd) + b_bd_1 - y, 2)
    return ret

def run(model_optimizer, 
        num_epochs, 
        aux_vars, 
        optim_vars,
        implicit_layer):
    '''
    runs the optimization loop for number of epochs while using implicit layers
    '''

    w1s, w2s = [optim_vars[0].numpy()[0]], [optim_vars[1].numpy()[0]]
    for epoch in tqdm(range(num_epochs)):
        model_optimizer.reset_grad()
        
        # retrieve params
        x, y, a = aux_vars
        w1, w2, b = optim_vars
        
        # get optimum value by solving the inner optimization problem
        b_star = implicit_layer(w1, w2, b)

        # compute loss for outer optimization
        loss = error_function(a, b_star, x, y) 
        numel = loss.shape[1]
        loss = ops.summation(loss)
        loss = ops.divide_scalar(loss, numel)
        
        # compute gradients 
        loss.backward()

        # modify params 
        model_optimizer.step()

        # store intermediate values of w1 and w2
        w1s.append(w1.numpy()[0])
        w2s.append(w2.numpy()[0])
    return w1s, w2s


class cost():
    def __init__(self):
        pass

class cost1(cost):
    '''
    custom cost function with forward and gradient defined (similar to Theseus)
    '''
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x**2

    def grad(self, x):
        return 2*x

class cost2(cost):
    '''
    custom cost function with forward and gradient defined (similar to Theseus)
    '''
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return (x+1)**2

    def grad(self, x):
        return 2*(x+1)


if __name__=='__main__':
    
    # generate sample data with some predefined params
    data_x, data_y, A, B, a, b  = generate_data(b=-0.5)
  

    # Plot the ground truth data
    fig, ax = plt.subplots()
    ax.scatter(data_x.numpy(), data_y.numpy(), label="Noisy Samples", lw=5)
    xs = np.linspace(data_x.numpy().min(), data_x.numpy().max())
    ax.plot(xs, a.numpy()*np.tanh(xs) + b, color='k', label="Ground Truth Function")
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title("Cost Function Weight Learning Ground Truth", fontsize=16)

    plt.savefig("results/weights_gt.png")
 

    
    np.random.seed(42)
    # Initialize  weights
    s_w1s = [2., 4., 8., 1.]
    s_w2s = [4., 2., 2., 8.]
    bstars = []
    for i in range(4):
        # wrap into needle trainable params
        w1 = Parameter(Tensor(np.array([s_w1s[i]]), requires_grad=True, device=ndl.cpu(), dtype="float32"))
        w2 = Parameter(Tensor(np.array([s_w2s[i]]), requires_grad=True, device=ndl.cpu(), dtype="float32"))


     
        b = Tensor(init.ones(*(1,), requires_grad=False, device=ndl.cpu(), dtype="float32"))*b
        a = Parameter(Tensor(init.ones(*(1,), requires_grad=True, device=ndl.cpu(), dtype="float32"))*2.)

        # set the variables to be optimised in the outer loop (aux vars)
        aux_vars = A, B, a
        
        # set the variables to be optimised in the inner loop (optim vars)
        optim_vars = w1, w2, b
  

        # set the outer loop optimizer with the appropiate params
        model_optimizer = ndl.optim.Adam([w1, w2, a], lr=1e-2, weight_decay=1e-3)

        # set the inner optimizer type
        opt = "Weights" 
  

        # define a custom weighted cost function from our baseline predefined cost functions
        cost_fn = [
                ndl.implicit_cost_function.NonLinearCostFunction(aux_vars, optim_vars, cost1()),
                ndl.implicit_cost_function.NonLinearCostFunction(aux_vars, optim_vars, cost2())
        ]


        # Initialize the implicit layer with the inner optim type, the cost function and the inner gradient calculation type
        implicit_layer = ndl.nn.WeightImplicitLayer(opt, cost_fn, "implicit")

        num_epochs = 1000
        
        # run the optimization for a given number of epochs
        w1s, w2s = run(model_optimizer, num_epochs, aux_vars, optim_vars, implicit_layer)
        
        # compute known optimal solution value given w1 and w2 (manually defined). This value is only used for plotting purposes 
        # to show our w1 and w2 do indeed converge b_star to its ground truth optimum value
        bstar = -np.array(w1s)/(np.array(w1s) + np.array(w2s))
        bstars.append(bstar)
   

    # generate the plots of convergence of b_star with different intializations
    generate_plots(bstars)


 
    


