#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import math
import matplotlib.pyplot as plt


class Dense_Layer(nn.Module):
    def __init__(self, in_channels, growthrate, bn_size):
        super(Dense_Layer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growthrate, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bn_size * growthrate)
        self.conv2 = nn.Conv2d(
            bn_size * growthrate, growthrate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, prev_features):
        out1 = torch.cat(prev_features, dim=1)
        out1 = F.relu(out1)
        out1 = self.conv1(out1)
        out1 = F.relu(out1)
        out2 = self.conv2(out1)
        return out2

class Dense_Block(nn.ModuleDict):
    def __init__(self, n_layers, in_channels, growthrate, bn_size):
        """
        A Dense block consists of `n_layers` of `Dense_Layer`
        Parameters
        ----------
            n_layers: Number of dense layers to be stacked 
            in_channels: Number of input channels for first layer in the block
            growthrate: Growth rate (k) as mentioned in DenseNet paper
            bn_size: Multiplicative factor for # of bottleneck layers
        """
        super(Dense_Block, self).__init__()

        layers = dict()
        for i in range(n_layers):
            layer = Dense_Layer(in_channels + i * growthrate, growthrate, bn_size)
            layers['dense{}'.format(i)] = layer
        
        self.block = nn.ModuleDict(layers)
    
    def forward(self, features):
        if(isinstance(features, torch.Tensor)):
            features = [features]
        
        for _, layer in self.block.items():
            new_features = layer(features)
            features.append(new_features)
        
        return torch.cat(features, dim=1)


class IteDNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(IteDNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.dense_block = nn.Sequential(
            Dense_Block(6, 32, growthrate=32, bn_size=4),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(224, 3, 3, 1, 1),
            )

    def forward(self, input):

        x = input

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
        
            x = self.dense_block(x)
            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, x_list
    

class IReDNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(IReDNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.dense_block = nn.Sequential(
            Dense_Block(6, 32, growthrate=32, bn_size=4),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(224, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)
            x = h

            x = self.dense_block(x)

            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, x_list
    


class IReDNet_LSTM(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(IReDNet_LSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.dense_block = nn.Sequential(
            Dense_Block(6, 32, growthrate=32, bn_size=4),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(224, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)
            x = h

            x = self.dense_block(x)

            x = self.conv(x)
            x_list.append(x)
        return x, x_list
    

class IReDNet_GRU(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(IReDNet_GRU, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_z = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.dense_block = nn.Sequential(
            Dense_Block(6, 32, growthrate=32, bn_size=4),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(224, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            x1 = torch.cat((x, h), 1)
            z = self.conv_z(x1)
            b = self.conv_b(x1)
            s = b * h
            s = torch.cat((s, x), 1)
            g = self.conv_g(s)
            h = (1 - z) * h + z * g

            x = h

            x = self.dense_block(x)
            x = self.conv(x)
            x_list.append(x)
        return x, x_list
    

class IReDNet_BiRNN(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(IReDNet_BiRNN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.dense_block = nn.Sequential(
            Dense_Block(6, 32, growthrate=32, bn_size=4),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(224, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h_forward = Variable(torch.zeros(batch_size, 32, row, col))
        h_backward = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h_forward = h_forward.cuda()
            h_backward = h_backward.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            # Forward pass
            x_forward = torch.cat((x, h_forward), 1)
            h_forward = self.conv_forward(x_forward)

            # Backward pass
            x_backward = torch.cat((x, h_backward), 1)
            h_backward = self.conv_backward(x_backward)

            # Combining forward and backward passes
            x = h_forward + h_backward

            x = self.dense_block(x)
            x = self.conv(x)
            x_list.append(x + input)  # Adding skip connection

        return x, x_list
    

class IReDNet_ConvLSTM(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(IReDNet_ConvLSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        # Initial convolutional layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        # ConvLSTM Layers
        self.convLSTM_i = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.convLSTM_f = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.convLSTM_c = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.convLSTM_o = nn.Conv2d(32 + 32, 32, 3, 1, 1)

        # Dense Block
        self.dense_block = Dense_Block(6, 32, growthrate=32, bn_size=4)

        # Output convolutional layer
        self.conv = nn.Conv2d(224, 3, 3, 1, 1)

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            combined = torch.cat((x, h), 1)

            # ConvLSTM calculations
            i = torch.sigmoid(self.convLSTM_i(combined))
            f = torch.sigmoid(self.convLSTM_f(combined))
            o = torch.sigmoid(self.convLSTM_o(combined))
            c_tilde = torch.tanh(self.convLSTM_c(combined))
            c = f * c + i * c_tilde
            h = o * torch.tanh(c)

            x = h
            x = self.dense_block(x)
            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, x_list
    


class IReDNet_IndRNN(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(IReDNet_IndRNN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        # Initial convolutional layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        # IndRNN layers
        self.recurrent_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32 + 32, 32, 3, 1, 1),  # Convolution for each IndRNN cell
                nn.ReLU()
            ) for _ in range(self.iteration)
        ])

        # Dense Block
        self.dense_block = nn.Sequential(
            Dense_Block(6, 32, growthrate=32, bn_size=4),
        )

        # Final convolutional layer
        self.conv = nn.Sequential(
            nn.Conv2d(224, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        if self.use_GPU:
            h = h.cuda()

        x_list = []
        for i in range(self.iteration):
            combined = torch.cat((input, x), 1)
            combined = self.conv0(combined)
            combined = torch.cat((combined, h), 1)
            h = self.recurrent_layers[i](combined)

            x = h

            x = self.dense_block(x)
            x = self.conv(x)
            x_list.append(x)
        return x, x_list
    

    
class IReDNet_QRNN(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(IReDNet_QRNN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        # Initial convolutional layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        # QRNN layers
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_z = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )

        # Dense Block
        self.dense_block = nn.Sequential(
            Dense_Block(6, 32, growthrate=32, bn_size=4),
        )

        # Final Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(224, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()

        x_list = []
        for _ in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            combined = torch.cat((x, h), 1)
            f = self.conv_f(combined)
            z = self.conv_z(combined)
            o = self.conv_o(combined)

            h = (1 - z) * h + z * o

            x = self.dense_block(h)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list

