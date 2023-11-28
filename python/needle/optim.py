"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

class LU(Optimizer):

    def __init__(self, device)-> None:
        """
        Based on the device picks cuda or cpu version of LU
        """
        pass
    
    def step(self):
        """
        """
        pass

class GN(Optimizer):
    def __init__(self, device) -> None:
        '''
        '''
        pass
    
    def step(self):
        """
        """
        pass

class InnerOptimizer(Optimizer):
    """
        This class takes a non linear solver which is used to iteratively perform 
        gradient steps for the inner NLLS problem
        The LinearSolver is used to solve the linearized version at each step 
        If NonLinearSolver is None, then assume the problem is linear
    """
    def __init__(self, device, NonlinearSolver=None, LinearSolver=None) -> None:
        self.NonlinearSolver=NonlinearSolver
        self.LinearSolver = LinearSolver
        self.device = device
    
    def solve(self, cost_fn):
        print('lmao peepeepoopoo')
        return 1.0