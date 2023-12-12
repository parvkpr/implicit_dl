"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        temp = PowerScalar(self.scalar-1)(node.inputs[0])
        temp2 = MulScalar(self.scalar)(temp)
        temp3 = EWiseMul()(out_grad, temp2)
        return temp3 
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        dividend, divisor = node.inputs
        reciprocal_divisor = PowerScalar(-1)(divisor)
        grad_dividend = EWiseMul()(out_grad, reciprocal_divisor)
        reciprocal_divisor_square = PowerScalar(2)(reciprocal_divisor)
        negation_dividend = Negate()(dividend)
        layer_divisor_grad = EWiseMul()(negation_dividend, reciprocal_divisor_square)
        grad_divisor = EWiseMul()(out_grad, layer_divisor_grad)

        return grad_dividend, grad_divisor
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        rec_scalar = Tensor(1/self.scalar, device=node.inputs[0].device)
        layer_grad = BroadcastTo(node.shape)(rec_scalar)
        grad_divscalar = EWiseMul()(out_grad, layer_grad)
        return grad_divscalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        b = numpy.arange(0, a.ndim)
        if self.axes:
          b[self.axes[0]],  b[self.axes[1]] = b[self.axes[1]],  b[self.axes[0]]
        else:
          b[-2], b[-1] = b[-1], b[-2]
        return a.permute(b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        #layer_grad = Tensor(array_api.ones(a.shape))
        layer_grad_1 = Transpose(self.axes)(out_grad)
        #layer_grad_2 = EWiseMul()(layer_grad_1 ,layer_grad)
        return layer_grad_1
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        #layer_grad = Tensor(array_api.ones(a.shape))
        new_out_grad = Reshape(a.shape)(out_grad)
        #layer_grad_1 = EWiseMul()(new_out_grad ,layer_grad)
        return new_out_grad
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print('this \n')
        # print(out_grad)
        a = node.inputs[0]
        input_shape = a.shape[::-1]
        output_shape = out_grad.shape[::-1]
        comp_list = []
        if out_grad.shape!=a.shape: 
          for i in range(len(output_shape)):
            if i < len(input_shape):
              if input_shape[i] != output_shape[i]:
                comp_list.append(len(output_shape)-i-1)
            else:
              comp_list.append(len(output_shape)-i-1)
          comp = tuple(comp_list)
          out_grad_1 = Summation(comp)(out_grad)
        else:
          out_grad_1 = out_grad

        #sum_grad = Tensor(array_api.ones(a.shape))
        out_grad_2 = Reshape(a.shape)(out_grad_1)
        return out_grad_2
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs
        #sum_grad = Tensor(numpy.ones(a[0].shape))
        goal_shape = out_grad.shape
        
        new_axes = list(out_grad.shape)
        if self.axes is not None:
          temp_axes = (self.axes, )
          for c in temp_axes:
            new_axes.insert(c, 1)
        # print(a)
        # print(out_grad.shape)
        # print(new_grad.shape)
        # print(sum_grad.shape)
        new_grad = Reshape(new_axes)(out_grad)
        out_grad_1 = BroadcastTo(a[0].shape)(new_grad)
        #sum_grad_1 = EWiseMul()(new_grad ,sum_grad)
        #print(sum_grad_1)
        return out_grad_1
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        rhs_trans = Transpose()(rhs)
        grad_lhs = MatMul()(out_grad, rhs_trans)
        lhs_trans =   Transpose()(lhs)
        grad_rhs = MatMul()(lhs_trans, out_grad)

        #this part is to deal with broad cast gradients when it happens in matmul
        comp_list_1 = []
        comp_list_2 = []
        input_shape = rhs.shape[::-1]
        output_shape = grad_rhs.shape[::-1]
        if grad_rhs.shape!=rhs.shape: 
          for i in range(len(output_shape)):
            if i < len(input_shape):
              if input_shape[i] != output_shape[i]:
                comp_list_1.append(len(output_shape)-i-1)
            else:
              comp_list_1.append(len(output_shape)-i-1)
          comp = tuple(comp_list_1)
          fin_grad_rhs = Summation(comp)(grad_rhs)
        else:
          fin_grad_rhs = grad_rhs
        
        input_shape = lhs.shape[::-1]
        output_shape = grad_lhs.shape[::-1]
        if grad_lhs.shape!=lhs.shape: 
          for i in range(len(output_shape)):
            if i < len(input_shape):
              if input_shape[i] != output_shape[i]:
                comp_list_2.append(len(output_shape)-i-1)
            else:
              comp_list_2.append(len(output_shape)-i-1)
          comp_2 = tuple(comp_list_2)
          fin_grad_lhs = Summation(comp_2)(grad_lhs)
        else:
          fin_grad_lhs = grad_lhs
        

        return fin_grad_lhs, fin_grad_rhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a*-1
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        #layer_grad = Tensor(array_api.ones(a.shape))
        layer_grad_2 = Negate()(out_grad)
        #layer_grad_2 = EWiseMul()(out_grad ,layer_grad_1)
        return layer_grad_2
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        layer_grad = PowerScalar(-1)(node.inputs[0])
        out_grad_1 = EWiseMul()(out_grad, layer_grad)
        return out_grad_1
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        layer_grad = Exp()(node.inputs[0])
        out_grad_1 = EWiseMul()(out_grad, layer_grad)
        return out_grad_1
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # c = node.inputs[0]
        # # print(c.shape)
        # prod=1
        # for i in range(len(c.shape)):
        #     prod*=c.shape[i]
        # d = c.reshape((prod,))
        # #d = c.flatten()
        # d = [0 if x < 0 else 1 for x in d]
        # eps = d.reshape((node.inputs[0].shape))
        # layer_grad = Tensor(eps)
        # out_grad_1 = EWiseMul()(out_grad, layer_grad)
        #data = ReLU()(node.inputs[0])
        #node > 0.0
        # grad = array_api.__ge__(data)
        #grad = numpy.zeros(data.shape, dtype=out_grad.dtype)
        #data >= 0
        #grad = data > 0
        #return out_grad*Tensor(grad, dtype=node.inputs[0].data.dtype, device=node.inputs[0].device)
        # return out_grad_1  
        # a = out_grad.realize_cached_data()
        #array_api.ewise
        c = node.inputs[0].realize_cached_data()
        d = c.flatten()
        d = [0 if x < 0 else 1 for x in d]
        eps = array_api.array(d).reshape(node.inputs[0].shape)
        layer_grad = Tensor(eps)
        out_grad_1 = EWiseMul()(out_grad, layer_grad)
        return out_grad_1   
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        layer_grad_1 = Exp()(node.inputs[0])
        layer_grad_2 = Exp()(Negate()(node.inputs[0]))
        layer_grad_3 = (layer_grad_1 + layer_grad_2)
        layer_grad_4 = PowerScalar(2)(layer_grad_3)
        layer_grad_5 = layer_grad_4/4
        layer_grad_6 = PowerScalar(-1)(layer_grad_5)
        
        out_grad_1 = EWiseMul()(out_grad, layer_grad_6)
        return out_grad_1
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        a = args[0]
        nshape = list(a.shape)
        total_num = len(args)
        nshape.insert(self.axis, total_num)
        empty = numpy.ones(shape=nshape)
        c = array_api.array(empty, device=args[0].device)
        temp_slice_list = nshape
        for i in range(len(temp_slice_list)):
            if i!=self.axis:
                temp_slice_list[i] = slice(None)
            else:
                temp_slice_list[i] = slice(0, a.shape[i], 1)
        start = 0
        stop = 1
        for i in range(len(args)):
            temp_slice_list[self.axis] = slice(start, stop, 1)
            #print(temp_slice_list)
            c[tuple(temp_slice_list)] = args[i]
            start = stop
            stop = stop+1
        return c
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print(node.inputs[0])
        # print(node.inputs)
        new_list = Split(self.axis)(out_grad)
        # print(new_list[0])
        # print(new_list[1])
        # print(len(new_list))
        # # print(new_list[0])
        # temp_out = new_list
        # print(len(a))
        # for i in range(len(a)):
        #     print(a[i].shape)
        # for i in range(len(new_list)):
        #     temp_out[i] = EWiseMul()(new_list[i], a[i])
        return new_list
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        #print('came till here')
        total_num = A.shape[self.axis]
        fin_list = []
        nshape = list(A.shape)
        nshape.pop(self.axis)
        start =0
        stop =1
        temp_slice_list = list(A.shape)
        for i in range(len(temp_slice_list)):
            if i!=self.axis:
                temp_slice_list[i] = slice(None)
            else:
                temp_slice_list[i] = slice(0, 1, 1)
        
        for i in range(total_num):
            empty = numpy.ones(shape=nshape)
            c = array_api.array(empty, device=A.device)
            d = c.shape
            # print(c.shape)
            temp_slice_list[self.axis] = slice(start, stop, 1)
            c = A[tuple(temp_slice_list)].compact()
            #c.compact() # if i dont call compact it fails
            e = c.reshape(nshape)
            # print(e.shape)
            fin_list.append(e)
            start = stop
            stop = stop+1
        
        return tuple(fin_list)



        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Stack(self.axis)(out_grad)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        c =  Flip(self.axes)(out_grad)
        #c.compact()
        return c
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        
        for axis in self.axes:
            if axis >= len(a.shape):
                return a
            new_shape[axis] = a.shape[axis]  + (a.shape[axis])*self.dilation 

        c = array_api.full(tuple(new_shape), 0, device=a.device)
        slices = [slice(None)] * len(new_shape)
        
        for axis in self.axes:
             slices[axis] = slice(None, None, self.dilation+1)

        c[tuple(slices)]= a

        return c
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return UnDilate(self.axes,self.dilation)(out_grad)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                return a
            new_shape[axis] = (a.shape[axis])//(1+self.dilation) 
        c = array_api.full(new_shape, 0, device=a.device)
        slices = [slice(None)] * len(new_shape)
        for axis in self.axes:
             slices[axis] = slice(None, None, self.dilation+1)
        d = a[tuple(slices)]
        d.compact()
        c = d
        return c
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        pad_ax = ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0))
        C = A.pad(pad_ax)
        Ns, Hs, Ws, Cs = C.strides
        #print(self.stride)
        # print(C.strides)
        print('pre')
        print(self.padding)
        print(C.shape)
        print(A.shape)
        # print(B.shape)
        # print('done') 
        #(N,H-K+1,W-K+1,C_out)
        # out = np.zeros((N,H-K+1,W-K+1,C_out))
        #out = array_api.full((N,H+(2*self.padding) -K+1,W+(2*self.padding)-K+1,C_out), 0, device=A.device)
        inner_dim = K * K * C_in
        height = H+(2*self.padding)-K+1
        width = H+(2*self.padding)-K+1
        if self.stride>1:
            height = height//self.stride
            width = width//self.stride
        C = C.as_strided(shape = (N, height, width, K, K, C_in),
                                        strides = (Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs))
        C = C.compact()
        
        E = C.reshape((C.shape[0]*C.shape[1]*C.shape[2],inner_dim))
        # print(E.shape)
        # print(E.strides)
        E = E.compact()
        # print('here')
        B = B.reshape((inner_dim, C_out))
        F = E @ B
        # print(F.shape)
        F = F.reshape((C.shape[0],C.shape[1], C.shape[2], C_out))
        # print(F.shape)
        return F
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


class Inverse(TensorOp):
    def compute(self, a):
        return array_api.inverse(a)

    def gradient(self, out_grad, node):
        return NotImplementedError("PLEASE DON'T NEED THIS")


def inverse(a):
    return Inverse()(a)


class LSImplicit(TensorOp):
    def __init__(self, opt, implicit_grad_method):
        self.opt = opt
        self.implicit_grad_method = implicit_grad_method

    def implicitDiff(self):
        # This computes partial b* / partial a
        if(self.opt == 'Scalar'):
            _, x, _ = self.cost_fn.aux_vars
            implicit_grad = - summation(x**2)/x.shape[1]
            implicit_grad = implicit_grad.detach()
            return implicit_grad
        elif(self.opt == 'Linear'):
            _, A, B = self.cost_fn.aux_vars
            _A, _B = A.numpy(), B.numpy()

            #TODO Support this in needle
            import numpy as np
            implicit_grad = np.linalg.inv(_A.T @ _A) @ _A.T #@ _B
            return Tensor(init.ones(*A.shape), device=A.device)
        else:
            raise NotImplementedError("Select Scalar or Linear opt")

    
    def unrollDiff(self):
        raise NotImplementedError("Unrolling is not implemented")
    
    def compute_grad(self):
        if self.implicit_grad_method == "implicit":
            return self.implicitDiff()
        elif self.implicit_grad_method == "unroll":
            return self.unrollDiff()
        
    def compute(self, x, y, A, B):
        '''
            A, B are data matrics
            x, y are variables to optimize

            y is optimized in the outer loop
            x is optimized here to find x_star.
        '''
        if(self.opt == 'Scalar'):
            ###
            # Solving Ax^2 + b = y
            ###
            x_b = x.reshape((1,1)).broadcast_to(A.shape)
            left_side = x_b * (A**2)
            right_side = B

        elif(self.opt == 'Linear'):
            ###
            # Solving Ay + x = B
            # Solve A^TAy = A^T(B - x)
            ###
            # Left side of equation
            left_side = A@A.permute((1,0))

            # Right side of equation
            b_minus_x = B - x.reshape((1,1)).broadcast_to(B.shape)
            right_side = A@b_minus_x.permute((1,0))


        # Solve equation
        self.x_star = array_api.solve(y, left_side, right_side, self.opt)
        return self.x_star

    def gradient(self, out_grad, node):
        '''
            We are only interested in the gradient for our variable of interest and so only
            that gradient is officially supported through this implementation
        '''
        cur_input = node.inputs[0]

        if(self.opt == 'Scalar'):
            out_grads = [Tensor(init.zeros(*ipt.shape), device=out_grad.device) for ipt in node.inputs]
            return out_grads

        elif(self.opt == 'Linear'):
            '''
                Compute partial x_star/partial y
            '''
            out_grads = [Tensor(init.zeros(*ipt.shape), device=out_grad.device) for ipt in node.inputs]
            return out_grads

        return summation(out_grad.reshape((1,out_grad.shape[0])) @ layer_grad)
        
def lsimplicit(inner_optimizer, implicit_grad_method, x, y, A, B):
    return LSImplicit(inner_optimizer, implicit_grad_method)(x, y, A, B)


class WeightLSImplicit(TensorOp):
    def __init__(self, opt, cost_fn, implicit_grad_method):
        self.opt = opt
        self.cost_fn = cost_fn
        self.implicit_grad_method = implicit_grad_method


    def compute(self, x, w1, w2):
        '''
            Solving x_star = w1*c1(x) + w2*c2(x)
            Finds minimum of both weighted cost functions using gradient descent

            Gradient descent is done here istead of array_api.solve because our gradient
            is calculated with cost function evaluations of x_star, which does not match
            the form for our other cases.
        '''
        self.x_star = x
        ws = [w1, w2]
        for i in range(100):
            grad1 = self.cost_fn[0].grad(self.x_star)
            grad2 = self.cost_fn[1].grad(self.x_star)
            self.x_star = self.x_star - 0.01*(w1*grad1 + w2*grad2)
        return self.x_star

    
    def gradient(self, out_grad, node):
        '''
            Implicit gradient of weights and data.
            Implicit gradients are specific to the input cost functions
            grad_w1 = -x_star/(w1+w2)
            grad_w2 = -(x_star+1)/(w1+w2)
        '''
        gradw1 = negate(power_scalar(node.inputs[1] * node.inputs[2], -1)) * self.x_star
        gradw2 = negate(power_scalar(node.inputs[1] * node.inputs[2], -1)) * (self.x_star+1)
        return multiply(out_grad, node.inputs[0]), \
               multiply(out_grad, gradw1), \
               multiply(out_grad, gradw2)
               


def weight_lsimplicit(inner_optimizer, cost_fn, implicit_grad_method, x, w1, w2):
    return WeightLSImplicit(inner_optimizer, cost_fn, implicit_grad_method)(x, w1, w2)
