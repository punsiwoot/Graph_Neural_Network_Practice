
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv
torch.manual_seed(1000)
np.random.seed(1000)

class simple_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_class, activation_fcn = "tanh"):
        super(simple_GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)   # hidden_channels is like a hidden layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels ,num_class) # for classifire a value from embeding space
        #activation function
        if activation_fcn == "tanh":
            self.activation_fcn = torch.nn.Tanh()
        elif activation_fcn == "relu" :
            self.activation_fcn = torch.nn.ReLU()
        else : self.activation_fcn = torch.nn.Tanh()
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation_fcn(x)
        x = self.conv2(x, edge_index)
        x = self.activation_fcn(x)
        
        out = self.classifier(x)
        return out,x