"""cost function module"""
from typing import Any
import needle as ndl
import numpy as np


class CostFunction():
    def __init__(self, aux_vars, optim_vars, cost_function):
        self.aux_vars = aux_vars
        self.optim_vars = optim_vars 
        self.cost_function = cost_function

class LinearCostFunction(CostFunction):
    """ 
        This class defines a linear inner loop objective in the form sum_i ||w(phi) (A_i x - b)||_2 
    """
    def __init__(self, aux_vars, optim_vars, cost_function):
        super().__init__(aux_vars, optim_vars, cost_function)

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        ###
        # TODO: Implment linear cost function
        ###
        #return ndl.ops.summation(self.optim_vars @ x - self.aux_vars[0], axis=(1,))
        return np.linalg.norm((self.optim_vars.numpy() @ x - self.aux_vars[0]).numpy())
 


class NonLinearCostFunction(CostFunction):
    """ 
        This class defines a non linear inner loop objective in the form sum_i || f_i(x) - z||_w(phi)
        where f_i is a non linear function 
    """
    def __init__(self, aux_vars, optim_vars, cost_function):
        super().__init__(aux_vars, optim_vars, cost_function)

    def __call__(self, a, *args: Any, **kwds: Any) -> Any:
        return self.cost_function(a)

    #def grad(self, a, *args: Any, **kwds: Any) -> Any:
    def grad(self, a, *args: Any, **kwds: Any) -> Any:
        return self.cost_function.grad(a)

