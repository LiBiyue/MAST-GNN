import torch

import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.mlp = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        return self.mlp(x)
class DiffGraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate, adj_len, orders=2, enable_bias=True):
        super(DiffGraphConv, self).__init__()
        self.in_channels = in_channels*(1+orders*adj_len)
        self.out_channels = out_channels
        self.enable_bias = enable_bias
        self.orders = orders
        self.drop_rate = drop_rate

        self.linear = Linear(self.in_channels, out_channels)

    def forward(self, x, adj_mats):
        output = [x]
        for adj in adj_mats:
            x_mul = torch.einsum('mn,bsni->bsmi', adj, x).contiguous()
            output.append(x_mul)
            for k in range(2, self.orders + 1):
                x_mul_k = torch.einsum('mn,bsni->bsmi', adj, x_mul).contiguous()
                output.append(x_mul_k)
                x_mul = x_mul_k

        x_gc = self.linear(torch.cat(output, dim=1))
        output = F.dropout(x_gc, self.drop_rate, training=self.training)
        return output


class OutputLayer(nn.Module):
    def __init__(self, skip_channels, end_channels, out_channels):
        super(OutputLayer, self).__init__()

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_channels, (1, 1), bias=True)

    def forward(self, x):
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x