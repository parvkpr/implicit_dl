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
def generate_data(num_points=100, x=[[1.]*50], y=0.1, noise_factor=0.01):
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
    #print(x_bd.shape, A.shape, y_bd.shape, B.shape)
    #traise
    ret = ops.power_scalar(x_bd @ A + y_bd - B, 2)
    return ret



import numpy as np
import matplotlib.pyplot as plt

def compute_norms(X, XGt, Y, YGt):
    norm_X = np.linalg.norm(X.numpy() - XGt, axis=1)
    norm_Y = np.linalg.norm(Y - YGt)#, axis=1)
    return norm_X, norm_Y

def generate_plots(data):
    nx = []
    ny = []
    sizes = []

    for X, XGt, Y, YGt in data:
        norm_x, norm_y = compute_norms(X, XGt, Y, YGt)
        nx.append(norm_x)
        ny.append(norm_y)
        sizes.append(X.shape[1])

    # Create plots
    #plt.figure(figsize=(10, 5))

    fig, ax = plt.subplots(figsize=(10,5), ncols=2)

    # Plot for X-XGt
    #plt.subplot(1, 2, 1)
    print()
    print()
    print()
    print()
    print(sizes, nx, ny)
    #for i in range(len(data)):
    ax[0].plot(sizes, nx, label=f'Data', lw=3, markersize=10, marker='o')
    ax[0].set_xlabel('Size of X', fontsize=14)
    ax[0].set_ylabel('Norm', fontsize=14)
    ax[0].set_title('Norm of X - XGt', fontsize=16)
    ax[0].set_ylim(0, 0.08)
    ax[0].legend()

    # Plot for Y-YGt
    #plt.subplot(1, 2, 2)
    #for i in range(len(data)):
    ax[1].plot(sizes, ny, label=f'Data', lw=3, markersize=10, marker='o')
    ax[1].set_xlabel('Size of X', fontsize=14)
    #ax[1].set_ylabel('Norm', fontsize=14)
    ax[1].set_title('Norm of Y - YGt', fontsize=16)
    ax[1].set_ylim(0, 0.04)
    ax[1].legend()

    # Show plots
    #plt.tight_layout()
    plt.savefig("./scaling.png")
    plt.savefig("./scaling.pdf")
    plt.show()

# Example usage
data = [
    (np.random.rand(10, 5), np.random.rand(10, 5), np.random.rand(10, 5), np.random.rand(10, 5)),
    # Add more data entries as needed
]



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
        loss = error_function(y, x_star, A, B) #.mean()
        numel = loss.shape[1]
        loss = ops.summation(loss)
        loss = ops.divide_scalar(loss, numel)
        loss.backward()
        model_optimizer.step()
    print("\nFinal y and x")
    print(y, x_star)
    return y, x_star

if __name__=='__main__':
    np.random.seed(137)
    data = []
    for NUM_X in [2, 5, 10, 20, 30]:
        print("\nNUM X: {}".format(NUM_X))
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

        model_optimizer = ndl.optim.Adam([y], lr=1e-1, weight_decay=1e-7)

        #opt = ndl.optim.InnerOptimizer(device='cpu')
        #opt = "Linear" # or Nonlinear
        opt = "Linear"
        cost_fn = ndl.implicit_cost_function.LinearCostFunction(aux_vars, 
                                                                optim_vars, 
                                                                error_function)
        implicit_layer = ndl.nn.ImplicitLayer(opt, cost_fn, "implicit")

        num_epochs = 10000

        final_y, final_x = run(model_optimizer, num_epochs, aux_vars, optim_vars, opt, cost_fn, implicit_layer)
        print("TARGET x and y")
        print(data_y, data_x)
        print("\nHEY LOOK MA WE MADE IT\n")
        print("Y ERROR: {}".format(np.linalg.norm(data_y - final_y.numpy())))
        print("X ERROR: {}".format(np.linalg.norm(data_x.numpy() - final_x.numpy())))

        #data.append(np.linalg.norm(data_x.numpy() - final_x.numpy()))
        data.append([data_x, final_x.numpy(), data_y, final_y.numpy()])

    print(data)
    generate_plots(data)


