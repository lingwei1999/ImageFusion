import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 1, dilation = 1, is_last=False):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        self.is_last = is_last

    def forward(self, x):
        out = self.conv2d(x)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out
        
# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


#  DenseNet network
class DenseNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(DenseNet, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        # decoder
        self.conv_de = ConvLayer(nb_filter[1]*2, nb_filter[1], kernel_size, stride)
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride, is_last = True)

    def forward(self, ir, vi):
        ir_en = self.conv1(ir)
        ir_en = self.DB1(ir_en)

        vi_en = self.conv1(vi)
        vi_en = self.DB1(vi_en)

        f = self.conv_de(torch.cat([ir_en, vi_en], dim=1))
        f_1 = self.conv2(f)
        f_2 = self.conv3(f_1)
        f_3 = self.conv4(f_2)
        output = self.conv5(f_3)

        return output


# Dense Block unit
class DenseBlock_half(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock_half, self).__init__()
        out_channels_def = 8
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


#  DenseNet_half network
class DenseNet_half(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(DenseNet_half, self).__init__()
        denseblock = DenseBlock_half
        nb_filter = [8, 32, 16, 8]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        # decoder
        self.conv_de = ConvLayer(nb_filter[1]*2, nb_filter[1], kernel_size, stride)
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride, is_last = True)

    def forward(self, ir, vi):
        ir_en = self.conv1(ir)
        ir_en = self.DB1(ir_en)

        vi_en = self.conv1(vi)
        vi_en = self.DB1(vi_en)

        f = self.conv_de(torch.cat([ir_en, vi_en], dim=1))
        f_1 = self.conv2(f)
        f_2 = self.conv3(f_1)
        f_3 = self.conv4(f_2)
        output = self.conv5(f_3)

        return output