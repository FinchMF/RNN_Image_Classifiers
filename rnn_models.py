import math

import torch
from torch import Tensor 

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torch.autograd import Variable



################################
# LINEAR FULLY CONNECTED LAYER #
################################

class Linear_lyer(nn.Module):
    """Implementation of Linear Fully Connectied Layer"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        """ Reset Weights"""
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 // math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input_):
        """ Forward Pass """
        x, y = input_.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = input_.matmul(self.weight.t())
        if self.bias is not None:
            output += self.bias
        out = output
        return out
    
    def extra_repr(self):
        """ Print Layer Configuration """
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

############
# GRU CELL #
############

class GRUcell(nn.Module):
    """Implementation of GRU Cell"""
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.input_to_hidden = Linear_lyer(input_size, 3 * hidden_size, bias=bias)
        self.hidden_to_hidden = Linear_lyer(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """ Reset Weights """
        std = 1.0 // math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        """Foward Pass"""

        x = x.view(-1, x.size(1))

        gate_x = self.input_to_hidden(x)
        gate_h = self.hidden_to_hidden(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


#####################
# RNN with GRU CELL #
#####################

class GRUmodel(nn.Module):
    """ Gated Recurrent Unit RNN Model """
    def __init__(self, input_dim, hidden_dim, n_layer, output_dim, bias=True):
        super(GRUmodel, self).__init__()

        self.hidden_dim = hidden_dim

        self.n_layer = n_layer

        self.gru = GRUcell(input_dim, hidden_dim, n_layer)

        self.fc = Linear_lyer(hidden_dim, output_dim)


    def forward(self, x):
        """ Forward Pass """
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.n_layer, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.n_layer, x.size(0), self.hidden_dim))
        
        outs = []

        hn = h0[0,:,:]

        for sequence in range(x.size(1)):
            hn = self.gru(x[:,sequence,:], hn)
            outs.append(hn)

        out = outs[-1].squeeze()

        output = self.fc(out)

        return output


#############
# LSTM CELL #
#############

class LSTMcell(nn.Module):
    """Implementation of LSTM Cell"""
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.input_to_hidden = Linear_lyer(input_size, 4 * hidden_size, bias=bias)
        self.hidden_to_hidden = Linear_lyer(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset Weights"""
        std = 1.0 // math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        """Forward Pass"""
        hx, cx = hidden

        x = x.view(-1, x.size(1))

        gates = self.input_to_hidden(x) + self.hidden_to_hidden(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, torch.tanh(cy))

        return (hy, cy)


######################
# RNN with LSTM CELL #
######################

class LSTMmodel(nn.Module):
    """Long Short Term Memory RNN Model """
    def __init__(self, input_dim, hidden_dim, n_layer, output_dim, bias=True):
        super(LSTMmodel, self).__init__()

        self.hidden_dim = hidden_dim
        
        self.n_layer = n_layer

        self.lstm = LSTMcell(input_dim, hidden_dim, n_layer)

        self.fc = Linear_lyer(hidden_dim, output_dim)

    def forward(self, x):
        """ Forward Pass """
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.n_layer, x.size(0), self.hidden_dim).cuda())
        else: 
            h0 = Variable(torch.zeros(self.n_layer, x.size(0), self.hidden_dim))
        
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.n_layer, x.size(0), self.hidden_dim).cuda())
        else: 
            c0 = Variable(torch.zeros(self.n_layer, x.size(0), self.hidden_dim))

        outs = []

        cn = c0[0,:,:]
        hn = h0[0,:,:]

        for sequence in range(x.size(1)):
            hn, cn = self.lstm(x[:,sequence,:], (hn,cn))
            outs.append(hn)

        out = outs[-1].squeeze()

        output = self.fc(out)

        return output

    