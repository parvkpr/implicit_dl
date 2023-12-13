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
        for i, _ in enumerate(self.params):
            if i not in self.u:
                self.u[i] = ndl.init.zeros_like(self.params[i]).detach()
            if self.params[i].requires_grad:
                T = (self.params[i].grad + ndl.mul_scalar(self.params[i], self.weight_decay)).detach()
                self.u[i] = (self.momentum*self.u[i] + (1-self.momentum)*T).detach()
                self.params[i].data = self.params[i].data - self.lr*self.u[i]
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
        self.t+=1
        for i, _ in enumerate(self.params):
            if i not in self.m:
                self.m[i] = ndl.init.zeros_like(self.params[i])
            if i not in self.v:
                self.v[i] = ndl.init.zeros_like(self.params[i])
            if self.params[i].requires_grad:
                T1 = self.params[i].grad + ndl.mul_scalar(self.params[i], self.weight_decay).detach()
                T2 = ndl.power_scalar(T1, 2).detach()
                self.m[i] = (self.beta1*self.m[i] + (1-self.beta1)*T1).detach()
                self.v[i] = (self.beta2*self.v[i] + (1-self.beta2)*T2).detach()
                
                mhat = self.m[i]/(1-self.beta1**self.t)
                vhat = self.v[i]/(1-self.beta2**self.t)

                self.params[i].data = (self.params[i].data - self.lr*ndl.divide(mhat, ndl.power_scalar(vhat, 1/2) + self.eps)).detach()
        
        ### END YOUR SOLUTION
