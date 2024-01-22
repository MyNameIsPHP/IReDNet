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
from torch.autograd import Variable


class Light_Dense_Layer(nn.Module):
    def __init__(self, in_channels, growthrate):
        super(Light_Dense_Layer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, growthrate // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growthrate // 2)
        self.conv2 = nn.Conv2d(growthrate // 2, growthrate, kernel_size=3, padding=1, bias=False)

    def forward(self, prev_features):
        out1 = torch.cat(prev_features, dim=1)
        out1 = F.relu(self.bn1(out1))
        out1 = self.conv1(out1)
        out1 = F.relu(self.bn2(out1))
        out2 = self.conv2(out1)
        return out2

class Light_Dense_Block(nn.Module):
    def __init__(self, n_layers, in_channels, growthrate):
        super(Light_Dense_Block, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(Light_Dense_Layer(in_channels + i * growthrate, growthrate))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, features):
        if isinstance(features, torch.Tensor):
            features = [features]
        
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        
        return torch.cat(features, dim=1)

class LightIteDNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LightIteDNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU()
            )
        self.dense_block = Light_Dense_Block(4, 16, growthrate=16)
        self.conv = nn.Conv2d(80, 3, 3, 1, 1)

    def forward(self, input):
        if self.use_GPU:
            input = input.cuda()

        x = input

        x_list = []
        for _ in range(self.iteration):
            combined = torch.cat((input, x), 1)
            combined = self.conv0(combined)
           
            x = self.dense_block(combined)

            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, x_list
    
class LightIReDNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LightIReDNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
            )
        self.dense_block = Light_Dense_Block(4, 16, growthrate=16)
        self.conv = nn.Conv2d(80, 3, 3, 1, 1)

    def forward(self, input):
        if self.use_GPU:
            input = input.cuda()

        x = input
        h = torch.zeros(input.size(0), 16, input.size(2), input.size(3), device=input.device)
        c = torch.zeros_like(h)

        x_list = []
        for _ in range(self.iteration):
            combined = torch.cat((input, x), 1)
            combined = self.conv0(combined)
            combined = torch.cat((combined, h), 1)
            i = self.conv_i(combined)
            f = self.conv_f(combined)
            g = self.conv_g(combined)
            o = self.conv_o(combined)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = self.dense_block(h)

            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, x_list
    

class LightIReDNet_LSTM(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LightIReDNet_LSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        # Initial Convolutional Layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU()
        )

        # LSTM Convolutional Layers
        self.conv_i = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
        )

        # Light Dense Block
        self.dense_block = Light_Dense_Block(4, 16, growthrate=16)

        # Final Convolution
        self.conv = nn.Conv2d(80, 3, 3, 1, 1)

    def forward(self, input):
        if self.use_GPU:
            input = input.cuda()

        x = input
        h = Variable(torch.zeros(input.size(0), 16, input.size(2), input.size(3), device=input.device))
        c = Variable(torch.zeros_like(h))

        x_list = []
        for _ in range(self.iteration):
            combined = torch.cat((input, x), 1)
            combined = self.conv0(combined)

            # LSTM forward pass
            combined = torch.cat((combined, h), 1)
            i = self.conv_i(combined)
            f = self.conv_f(combined)
            g = self.conv_g(combined)
            o = self.conv_o(combined)

            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h

            x = self.dense_block(x)

            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, x_list
    

class LightIReDNet_GRU(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LightIReDNet_GRU, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        # Initial Convolutional Layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU()
        )

        # GRU-like gates
        self.conv_z = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Tanh()
        )

        # Light Dense Block
        self.dense_block = Light_Dense_Block(4, 16, growthrate=16)

        # Final Convolution
        self.conv = nn.Conv2d(80, 3, 3, 1, 1)

    def forward(self, input):
        if self.use_GPU:
            input = input.cuda()

        x = input
        h = Variable(torch.zeros(input.size(0), 16, input.size(2), input.size(3)))
        if self.use_GPU:
            h = h.cuda()

        x_list = []
        for _ in range(self.iteration):
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
            x = x + input  # Adding skip connection
            x_list.append(x)

        return x, x_list
    

class LightIReDNet_BiRNN(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LightIReDNet_BiRNN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        # Initial Convolutional Layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU()
        )

        # Forward and backward layers
        self.conv_forward = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),
            nn.Tanh()
        )

        # Dense Block
        self.dense_block = Light_Dense_Block(4, 16, growthrate=16)

        # Final Convolution
        self.conv = nn.Conv2d(80, 3, 3, 1, 1)

    def forward(self, input):
        if self.use_GPU:
            input = input.cuda()

        x = input
        h_forward = torch.zeros(input.size(0), 16, input.size(2), input.size(3), device=input.device)
        h_backward = torch.zeros_like(h_forward)

        x_list = []
        for _ in range(self.iteration):
            combined = torch.cat((input, x), 1)
            combined = self.conv0(combined)

            # Forward pass
            combined_forward = torch.cat((combined, h_forward), 1)
            h_forward = self.conv_forward(combined_forward)

            # Backward pass
            combined_backward = torch.cat((combined, h_backward), 1)
            h_backward = self.conv_backward(combined_backward)

            # Combining forward and backward passes
            h_combined = h_forward + h_backward

            x = self.dense_block(h_combined)

            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, x_list



class LightIReDNet_IndRNN(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LightIReDNet_IndRNN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        # Initial Convolutional Layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU()
        )

        # IndRNN layers
        self.recurrent_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(16 + 16, 16, 3, 1, 1),  # Convolution for each IndRNN cell
                nn.ReLU()
            ) for _ in range(self.iteration)
        ])

        # Dense Block (assuming a compatible Dense Block implementation)
        self.dense_block = Light_Dense_Block(4, 16, growthrate=16)

        # Final Convolution
        self.conv = nn.Conv2d(80, 3, 3, 1, 1)

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 16, row, col))
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
    
class LightIReDNet_ConvLSTM(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LightIReDNet_ConvLSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        # Initial Convolutional Layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU()
        )

        # ConvLSTM Layers - reduced channel size for lightness
        self.convLSTM_i = nn.Conv2d(16 + 16, 16, 3, 1, 1)
        self.convLSTM_f = nn.Conv2d(16 + 16, 16, 3, 1, 1)
        self.convLSTM_c = nn.Conv2d(16 + 16, 16, 3, 1, 1)
        self.convLSTM_o = nn.Conv2d(16 + 16, 16, 3, 1, 1)

        # Light Dense Block
        self.dense_block = Light_Dense_Block(4, 16, growthrate=16)

        # Final Convolution
        self.conv = nn.Conv2d(80, 3, 3, 1, 1)

    def forward(self, input):
        if self.use_GPU:
            input = input.cuda()

        x = input
        h = torch.zeros(input.size(0), 16, input.size(2), input.size(3), device=input.device)
        c = torch.zeros_like(h)

        x_list = []
        for _ in range(self.iteration):
            combined = torch.cat((input, x), 1)
            combined = self.conv0(combined)

            # ConvLSTM calculations
            i = torch.sigmoid(self.convLSTM_i(torch.cat((combined, h), 1)))
            f = torch.sigmoid(self.convLSTM_f(torch.cat((combined, h), 1)))
            o = torch.sigmoid(self.convLSTM_o(torch.cat((combined, h), 1)))
            c_tilde = torch.tanh(self.convLSTM_c(torch.cat((combined, h), 1)))

            c = f * c + i * c_tilde
            h = o * torch.tanh(c)

            x = self.dense_block(h)
            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, x_list

class LightIReDNet_QRNN(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LightIReDNet_QRNN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        # Initial convolutional layer with reduced number of filters
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),  # Reduced filters from 32 to 16
            nn.ReLU()
        )

        # Simplified QRNN layers with fewer filters
        self.conv_f = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),  # Reduced filters from 32 to 16
            nn.Sigmoid()
        )
        self.conv_z = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),  # Reduced filters from 32 to 16
            nn.Sigmoid()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, 1),  # Reduced filters from 32 to 16
            nn.Tanh()
        )

        # Light Dense Block (assuming a similar structure to LightIReDNet_BiRNN)
        self.dense_block = Light_Dense_Block(4, 16, growthrate=16)

        # Final Convolution with adjusted number of input filters
        self.conv = nn.Conv2d(80, 3, 3, 1, 1)  # Adjusted input filters

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 16, row, col))  # Adjusted number of filters

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

            x = x + input  # Skip connection
            x_list.append(x)

        return x, x_list
