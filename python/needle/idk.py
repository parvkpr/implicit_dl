from ..autograd import Op, Tensor, Value, TensorOp

class NLLSGrad():
    def __init__(self,device, method):
        self.method = method 
        self.device = device 
    def implicitDiff(self, cost_fn, x_star):
        pass 
    def unrollDiff(self, cost_fn, x_star):
        pass 
    def compute_grad(self, cost_fn, x_star):
        if self.method == "implicit":
            return self.implicitDiff(cost_fn, x_star)
        elif self.method == "unroll":
            return self.unrollDiff(cost_fn, x_star)

     

class LU:

    def __init__(device):
        """
        Based on the device picks cuda or cpu version of LU
        """
        pass

class GN:
    def __init__(device) -> None:
        '''
        
        '''
        pass

class InnerOptimizer:
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
    
    def solve(self):
        pass 

class CostFunction():
    def __init__(self, weights):
        self.weights = weights 

class LinearCostFunction(CostFunction):
    """ 
        This class defines a linear inner loop objective in the form sum_i ||w(\phi) (A_i x - b)||_2 
    """
    def __init__(self):
        super.__init__()
 


class NonLinearCostFunction(CostFunction):
    """ 
        This class defines a non linear inner loop objective in the form sum_i ||w(\phi) f_i(x)||_2
        where f_i is a non linear function 
    """
    def __init__(self):
        super.__init__()


class LSImplicit(TensorOp):
    def __init__(self, opt:InnerOptimizer, cost_fn:CostFunction, nllsGrad:NLLSGrad):
        """
        """
        self.opt = opt
        self.cost_fn = cost_fn
        self.nllsGrad = nllsGrad

        
    def compute(self):
        self.x_star = self.opt.solve(self.cost_fn)
        return self.x_star 
    def gradient(self, out_grad, node):
        layer_grad = self.nllsGrad.compute_grad(self.cost_fn, self.x_star)
        return out_grad @ layer_grad
        
