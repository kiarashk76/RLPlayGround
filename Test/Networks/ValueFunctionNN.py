import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueFunctionNN(nn.Module):
    def __init__(self, inputs):
        super(ValueFunctionNN, self).__init__()
        batch, w, h, channel_in = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        # w, h = inputs.shape[0], inputs.shape[1]
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size=3, stride=1):
        #     return (size - (kernel_size - 1) - 1) // stride + 1
        #
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        linear_input_size = w * h * channel_in
        self.head = nn.Linear(linear_input_size, 1)

    def forward(self, x):
        batch, w, h, channel_in = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # w, h = x.shape[0], x.shape[1]

        # x = x.view(batch, channel_in, h, w)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten()
        return self.head(x)
        # return self.head(x.view(x.size(0)))
