import data
import os
import sys
import re
import matplotlib
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

# https://blog.csdn.net/zwqjoy/article/details/94750649
class ClickRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batch_size, bidirectional=True, drop_prob=0.5, train_on_gpu=False):
        super(ClickRNN, self).__init__()
         
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.train_on_gpu = train_on_gpu
        
        # self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True,
                            bidirectional=bidirectional)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # linear and sigmoid layers
        if bidirectional:
          self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
          self.fc = nn.Linear(hidden_dim, output_dim)
          
        self.sigmoid = nn.Sigmoid()
        
 
    def forward(self, x, hidden=None):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        
        # embeddings and lstm_out
#         x = x.long()
#         embeds = self.embedding(x)
        
        if hidden is None:
            hidden = self.init_hidden(self.train_on_gpu)
        
        lstm_out, hidden = self.lstm(x, hidden)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sigmoid(out)
        out = sig_out[:, -1, :]
        
        # return last sigmoid output and hidden state
        return out, hidden
    
    
    def init_hidden(self, train_on_gpu):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        number = 1
        if self.bidirectional:
           number = 2
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers*number, self.batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers*number, self.batch_size, self.hidden_dim).zero_().cuda()
                     )
        else:
            hidden = (weight.new(self.n_layers*number, self.batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers*number, self.batch_size, self.hidden_dim).zero_()
                     )
        
        return hidden