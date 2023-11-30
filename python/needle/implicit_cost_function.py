"""cost function module"""
import needle as ndl
import numpy as np


class CostFunction():
    def __init__(self, aux_vars, optim_vars, cost_function):
        self.aux_vars = aux_vars
        self.optim_vars = optim_vars 
        self.cost_function = cost_function

class LinearCostFunction(CostFunction):
    """ 
        This class defines a linear inner loop objective in the form sum_i ||w(\phi) (A_i x - b)||_2 
    """
    def __init__(self, aux_vars, optim_vars, cost_function):
        super().__init__(aux_vars, optim_vars, cost_function)
 


class NonLinearCostFunction(CostFunction):
    """ 
        This class defines a non linear inner loop objective in the form sum_i || f_i(x) - z||_w(\phi)
        where f_i is a non linear function 
    """
    def __init__(self, aux_vars, optim_vars):
        super().__init__(aux_vars, optim_vars)


