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
def generate_data(num_points=1000, a=1, b=0.5, noise_factor=0.01):
    # Generate data: 100 points sampled from the quadratic curve listed above
    data_x = init.rand(1, num_points, device=device)*2 - 1
    noise = init.randn(1, num_points, device=device) * noise_factor
    #data_y = a * data_x**2 + b + noise
    #data_y = a * data_x**3 + b + noise
    data_y = a * ops.tanh(data_x) + b + noise
    #data_y = a * ops.exp(data_x) + b + noise

    x = Tensor(data_x, device=ndl.cpu())
    y = Tensor(data_y, device=ndl.cpu())
    return data_x, data_y, x, y, Tensor(np.array([a]), device=ndl.cpu()), b

def error_function(a, b, x, y):
    #xsquare = ops.power_scalar(x, 2)
    #xsquare = ops.power_scalar(x, 3)
    xsquare = ops.tanh(x)
    #xsquare = ops.exp(x)

    a_bd = ops.broadcast_to(a.reshape((1,1)), xsquare.shape)
    b_bd_1 = ops.broadcast_to(b.reshape((1,1)), xsquare.shape)
    #print(type(a_bd))
    #print(type(xsquare))
    #ret = ops.power_scalar(xsquare*a_bd + b_bd_1 - y, 2)
    ret = ops.power_scalar(ops.EWiseMul()(xsquare, a_bd) + b_bd_1 - y, 2)
    #raise
    return ret

def run(model_optimizer, 
        num_epochs, 
        aux_vars, 
        optim_vars,
        opt,
        cost_fn,
        implicit_layer):

    w1s, w2s = [optim_vars[0].numpy()[0]], [optim_vars[1].numpy()[0]]
    for epoch in tqdm(range(num_epochs)):
        model_optimizer.reset_grad()
        x, y, a = aux_vars
        w1, w2, b = optim_vars
        b_star = implicit_layer(w1, w2, b)
        loss = error_function(a, b_star, x, y) #.mean()
        numel = loss.shape[1]
        loss = ops.summation(loss)
        loss = ops.divide_scalar(loss, numel)
        loss.backward()
        #print(w1, w2)
        model_optimizer.step()
        w1s.append(w1.numpy()[0])
        w2s.append(w2.numpy()[0])
    print("Final a and b")
    print(a, b_star)
    print("FINAL WEIGHTS:")
    print("W1: {}".format(w1))
    print("W2: {}".format(w2))
    return w1s, w2s


class cost():
    def __init__(self):
        pass

class cost1(cost):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x**2

    def grad(self, x):
        return 2*x

class cost2(cost):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return (x+1)**2

    def grad(self, x):
        return 2*(x+1)


if __name__=='__main__':
    #data_x, data_y, A, B, a, b  = generate_data(b=-1)
    data_x, data_y, A, B, a, b  = generate_data(b=-0.5)
    #data_x, data_y, A, B, a, b  = generate_data(b=-1)
    # Plot the data
    fig, ax = plt.subplots()
    ax.scatter(data_x.numpy(), data_y.numpy(), label="Noisy Samples", lw=5)
    xs = np.linspace(data_x.numpy().min(), data_x.numpy().max())
    ax.plot(xs, a.numpy()*np.tanh(xs) + b, color='k', label="Ground Truth Function")
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title("Cost Function Weight Learning Ground Truth", fontsize=16)
    #plt.savefig("./weights_gt.pdf")
    plt.savefig("./weights_gt.png")
    #plt.show()
    #raise

    # Initial weights
    print(np.random.rand(1))
    np.random.seed(42)
    s_w1s = [2., 4., 8., 1.]
    s_w2s = [4., 2., 2., 8.]
    bstars = []
    for i in range(4):
        #w1 = Parameter(Tensor(np.array([2.]), requires_grad=True, device=ndl.cpu(), dtype="float32"))
        #w2 = Parameter(Tensor(np.array([1.]), requires_grad=True, device=ndl.cpu(), dtype="float32"))
        print(s_w1s)
        w1 = Parameter(Tensor(np.array([s_w1s[i]]), requires_grad=True, device=ndl.cpu(), dtype="float32"))
        w2 = Parameter(Tensor(np.array([s_w2s[i]]), requires_grad=True, device=ndl.cpu(), dtype="float32"))
        print("INITIAL WEIGHTS: {}".format([w1,w2]))
        #raise

        #x = Tensor(init.ones(*(1,), requires_grad=False, device=ndl.cpu(), dtype="float32"))*5
        b = Tensor(init.ones(*(1,), requires_grad=False, device=ndl.cpu(), dtype="float32"))*b
        a = Parameter(Tensor(init.ones(*(1,), requires_grad=True, device=ndl.cpu(), dtype="float32"))*1.)
        #aux_vars = x, A, B
        aux_vars = A, B, a
        optim_vars = w1, w2, b
        #optim_vars2 = w2
        #optim_vars = [optim_vars1, optim_vars2]
        #raise

        #model_optimizer = ndl.optim.Adam([w1, w2], lr=1e-3, weight_decay=1e-3)
        model_optimizer = ndl.optim.Adam([w1, w2, a], lr=1e-2, weight_decay=1e-3)

        print(-w2/(w1+w2))

        #opt = ndl.optim.InnerOptimizer(device='cpu')
        #opt = "Linear" # or Nonlinear
        #opt = "Nonlinear" # or Nonlinear
        #opt = "Scalar" # or Nonlinear
        opt = "Weights" # or Nonlinear
        #opt = "None"
        #cost_fn = ndl.implicit_cost_function.LinearCostFunction(aux_vars, 
        #                                                        optim_vars, 
        #                                                        error_function)
        cost_fn = [
                ndl.implicit_cost_function.NonLinearCostFunction(aux_vars, optim_vars, cost1()),
                ndl.implicit_cost_function.NonLinearCostFunction(aux_vars, optim_vars, cost2())
        ]

        implicit_layer = ndl.nn.WeightImplicitLayer(opt, cost_fn, "implicit")

        num_epochs = 5000

        w1s, w2s = run(model_optimizer, num_epochs, aux_vars, optim_vars, opt, cost_fn, implicit_layer)
        bstar = -np.array(w1s)/(np.array(w1s) + np.array(w2s))
        bstars.append(bstar)
    print("\nHEY LOOK MA WE MADE IT\n")
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
    plt.savefig("w_convergence.png")
    plt.savefig("w_convergence.pdf")
    plt.show()

    #ax
    


