"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        temp_1 = ops.power_scalar((1 + ops.exp(-x)),-1)
        return temp_1
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = (1/hidden_size)**0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.nonlin = nonlinearity
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            y = np.zeros((X.shape[0], self.W_hh.shape[0]), dtype=X.dtype)
            h = Tensor(y, device=X.device, dtype=X.dtype)
      
        out_1 = X @ self.W_ih
        out_2 = h @ self.W_hh
        if self.bias_ih is not None:
            new_b = [1]*len(out_1.shape)
            new_b[len(new_b)-1] = self.bias_ih.shape[0]
            bias_2 = ops.broadcast_to(ops.reshape(self.bias_ih, tuple(new_b)), out_1.shape)
            out_f = out_1 + bias_2
            new_b_2 = [1]*len(out_2.shape)
            new_b_2[len(new_b_2)-1] = self.bias_hh.shape[0]
            bias_3 = ops.broadcast_to(ops.reshape(self.bias_hh, tuple(new_b)), out_2.shape)
            out_f_2 = out_2 + bias_3
        else:
            out_f=out_1
            out_f_2 = out_2
        f_out = out_f +out_f_2
        if self.nonlin=='tanh':
            f_out = ops.tanh(f_out)
        else:
            f_out = ops.relu(f_out)
        return f_out
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.rnn_cells = []
        for layer in range(num_layers):
            if layer == 0:
                cell_input_size = input_size
            else:
                cell_input_size = hidden_size
            cell = RNNCell(input_size=cell_input_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)
            self.rnn_cells.append(cell)
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            y = np.zeros((self.num_layers, X.shape[1], self.rnn_cells[0].W_hh.shape[0]), dtype=X.dtype)
            h0 = Tensor(y, device=X.device, dtype=X.dtype)

        h_t = h0
        output = []
        # print(X.shape)
        # print(h_t.shape)
        # print('done here')
        temp_in = ops.split(X, 0)
        # temp_h = ops.split(h_t, 0)
        # print(len(temp_in))
        # print(temp_in[0].shape)
        # print(X.shape)
        for t in range(X.shape[0]):
            temp_h = ops.split(h_t, 0)
            x_t = temp_in[t]
            h_t_next = []
            for layer in range(self.num_layers):
                cell = self.rnn_cells[layer]
                # print('do')
                # print(x_t.shape)
                # print(temp_h[layer].shape)
                if layer ==0:
                    last = cell(x_t, temp_h[layer])
                else:
                    last = cell(last, temp_h[layer])
                h_t_next.append(last)
            h_t = ops.stack(h_t_next, axis=0)
            output.append(last)  # Output from the last layer
        o_t = ops.stack(output, axis=0)
        return o_t, h_t
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        k = (1/hidden_size)**0.5
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            # If initial hidden state is not provided, initialize with zeros
            y = np.zeros((X.shape[0], self.hidden_size), dtype=X.dtype)
            h = Tensor(y, device=X.device, dtype=X.dtype)
            d = np.zeros((X.shape[0], self.hidden_size), dtype=X.dtype)
            c = Tensor(d, device=X.device, dtype=X.dtype)
        else:
            h, c = h

        # # Perform input-hidden and hidden-hidden transformations
        # #gates = combined @ self.W_ih + self.bias_ih + h @ self.W_hh + self.bias_hh
        out_1 = X @ self.W_ih
        out_2 = h @ self.W_hh
        if self.bias_ih is not None:
            new_b = [1]*len(out_1.shape)
            new_b[len(new_b)-1] = self.bias_ih.shape[0]
            bias_2 = ops.broadcast_to(ops.reshape(self.bias_ih, tuple(new_b)), out_1.shape)
            out_f = out_1 + bias_2
            new_b_2 = [1]*len(out_2.shape)
            new_b_2[len(new_b_2)-1] = self.bias_hh.shape[0]
            bias_3 = ops.broadcast_to(ops.reshape(self.bias_hh, tuple(new_b)), out_2.shape)
            out_f_2 = out_2 + bias_3
        else:
            out_f=out_1
            out_f_2 = out_2
        f_out = out_f +out_f_2
        input_gate, forget_gate, cell_gate, output_gate = ops.split(f_out.reshape((f_out.shape[0], 4, self.W_hh.shape[0])), axis=1)
        # # Apply activation functions
        # s = Sigmoid()()
        input_gate = Sigmoid()(input_gate)#s.forward(input_gate)
        forget_gate = Sigmoid()(forget_gate)
        cell_gate = ops.tanh(cell_gate)
        output_gate = Sigmoid()(output_gate)

        # # Update cell state
        c = forget_gate * c + input_gate * cell_gate

        # # Update hidden state
        h = output_gate * ops.tanh(c)

        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.lstm_cells = []
        self.hidden_size = hidden_size
        for layer in range(num_layers):
            if layer == 0:
                cell_input_size = input_size
            else:
                cell_input_size = hidden_size
            cell = LSTMCell(input_size=cell_input_size, hidden_size=hidden_size, bias=bias, device=device, dtype=dtype)
            self.lstm_cells.append(cell)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # if h is None:
        #     y = np.zeros((self.num_layers, X.shape[1], self.rnn_cells[0].W_hh.shape[0]), dtype=X.dtype)
        #     h0 = Tensor(y, device=X.device, dtype=X.dtype)
        if h is None:
            # If initial hidden state is not provided, initialize with zeros
            y = np.zeros((self.num_layers, X.shape[1], self.hidden_size), dtype=X.dtype)
            h = Tensor(y, device=X.device, dtype=X.dtype)
            d = np.zeros((self.num_layers,X.shape[1], self.hidden_size), dtype=X.dtype)
            c = Tensor(d, device=X.device, dtype=X.dtype)
        else:
            h, c = h
        h_t = h
        c_t = c
        output = []
        # print(X.shape)
        # print(h_t.shape)
        # print('done here')
        temp_in = ops.split(X, 0)
        # temp_h = ops.split(h_t, 0)
        # print(len(temp_in))
        # print(temp_in[0].shape)
        # print(X.shape)
        for t in range(X.shape[0]):
            temp_h = ops.split(h_t, 0)
            temp_c = ops.split(c_t, 0)
            x_t = temp_in[t]
            h_t_next = []
            c_t_next = []
            for layer in range(self.num_layers):
                cell = self.lstm_cells[layer]
                # print('do')
                # print(x_t.shape)
                # print(temp_h[layer].shape)
                # print(temp_c[layer].shape)
                if layer ==0:
                    tog = (temp_h[layer], temp_c[layer])
                    last_h, last_c = cell(x_t, h=tog)
                else:
                    # last = ops.stack()
                    last_h, last_c = cell(last_h, h=(temp_h[layer], temp_c[layer]))
                h_t_next.append(last_h)
                c_t_next.append(last_c)
            h_t = ops.stack(h_t_next, axis=0)
            c_t = ops.stack(c_t_next, axis=0)
            output.append(last_h)  # Output from the last layer
        o_t = ops.stack(output, axis=0)
        return o_t, (h_t, c_t)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.randn(num_embeddings,  embedding_dim, device=device, dtype=dtype, requires_grad=True))
        #self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        # Create one-hot vectors
        one_hot = init.one_hot(self.weight.shape[0], x, device=x.device, dtype=x.dtype, requires_grad=True)
        print(one_hot.shape)
        print(self.weight.shape)
        one_hot = one_hot.reshape((one_hot.shape[0]*one_hot.shape[1],one_hot.shape[2]))
        # Project one-hot vectors to embeddings
        print(one_hot.shape)
        output = one_hot @ self.weight
        output= output.reshape((x.shape[0], x.shape[1], self.weight.shape[1]))
        return output
        ### END YOUR SOLUTION