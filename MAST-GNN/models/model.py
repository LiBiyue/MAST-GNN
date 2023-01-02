import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import layer
from models.self_attention import self_attention


class AirspaceModel(nn.Module):
    def __init__(self,
                 in_channels=17,
                 out_channels=1,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=128,
                 supports=None,
                 drop_rate=0.3,
                 kernel_size=2,
                 blocks=4,
                 layers=4,
                 use_graph_conv=True,
                 adaptive_mat_init=None,
                 adaptive_mat_size=10,
                 handle_minor_features=False,
                 use_adaptive_mat_only=False,
                 geo_adj=None,
                 flow_adj=None,
                 device="cuda:0"):
        super(AirspaceModel, self).__init__()
        self.blocks = blocks
        self.layers = layers
        self.drop_rate = drop_rate
        self.handle_minor_features = handle_minor_features
        self.supports = supports or []
        self.supports_len = len(self.supports)
        self.use_graph_conv = use_graph_conv
        self.adaptive_mat_init = adaptive_mat_init
        self.use_adaptive_mat_only = use_adaptive_mat_only
        self.device = device
        receptive_field = 1
        depth = list(range(blocks * layers))
        if self.use_graph_conv and self.adaptive_mat_init is not None:
            nodevecs = self.svd_init(adaptive_mat_size, self.adaptive_mat_init)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [nn.Parameter(n.to(self.device), requires_grad=True) for n in nodevecs]
        # tmpadj = torch.mm(self.nodevec1, self.nodevec2)
        # adaadjnumpy = tmpadj.cpu().detach().numpy()
        # np.save("adaadjnumpy.npy", adaadjnumpy)
        geo_adj=(geo_adj-np.min(geo_adj))/(np.max(geo_adj)-np.min(geo_adj))
        flow_adj = (flow_adj - np.min(flow_adj)) / (np.max(flow_adj) - np.min(flow_adj))
        self.geo_adj = torch.tensor(geo_adj).to(self.device)
        self.flow_adj = torch.tensor(flow_adj).to(self.device)
        self.geo_adjweight = nn.Parameter(torch.randn((126, 1)).to(self.device), requires_grad=True)
        self.flow_adjweight = nn.Parameter(torch.randn((126, 1)).to(self.device), requires_grad=True)
        self.theta = nn.Parameter(torch.randn((126, 126)).to(self.device), requires_grad=True)

        # self.nn.AdaptiveAvgPool2d((17,17))
        # self.madapt=m(self.madapt.unsqueeze(dim=0)).squeeze()#17*17
        # self.softmaxforadapt=nn.Softmax(dim=1)

        self.supports_len = self.supports_len if not use_adaptive_mat_only else 1

        if self.handle_minor_features:
            self.start_conv = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))
            self.minor_features_conv = nn.Conv2d(in_channels - 1, residual_channels, kernel_size=(1, 1))
        else:
            self.start_conv = nn.Conv2d(in_channels, residual_channels, kernel_size=(1, 1))
        self.residual_convs = nn.ModuleList([nn.Conv2d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = nn.ModuleList([nn.Conv2d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = nn.ModuleList([nn.BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = nn.ModuleList(
            [layer.DiffGraphConv(dilation_channels, residual_channels, drop_rate, self.supports_len) for _ in depth])
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=dilation))
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=dilation))
                dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.output_layer = layer.OutputLayer(skip_channels, end_channels, out_channels)
        self.attention = self_attention(dc=17, dk=30, dv=17)
        self.finalFC = nn.Linear(in_features=17 * 2, out_features=1, bias=False)
        self.receptive_field = receptive_field

    @staticmethod
    def svd_init(adp_mat_size, adp_mat_init):
        m, p, n = torch.svd(adp_mat_init)
        nodevec1 = torch.mm(m[:, :adp_mat_size], torch.diag(p[:adp_mat_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:adp_mat_size] ** 0.5), n[:, :adp_mat_size].t())
        return nodevec1, nodevec2

    def forward(self, x):
        # Input shape is (batch_size, seq_len, n_vertex, features)
        batch_size, seq_len, n_vertex, features = x.shape
        x = x.transpose(1, 3)
        seq_len = x.size(3)
        if seq_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - seq_len, 0, 0, 0))

        if self.handle_minor_features:
            x = self.start_conv(x[:, -1, ...].unsqueeze(1)) + F.leaky_relu(self.minor_features_conv(x[:, :-1, ...]))
        else:
            x = self.start_conv(x)
        skip = 0
        # for node in range(n_vertex):
        #     for t in range(seq_len):
        #         for i in range():
        #             return
        eg = torch.sigmoid(torch.mm(self.geo_adj, self.geo_adjweight))
        ef = torch.sigmoid(torch.mm(self.flow_adj, self.flow_adjweight))
        alphagf = torch.trace(torch.mm(torch.mm(eg.t(), self.theta), ef))
        alphagg = torch.trace(torch.mm(torch.mm(eg.t(), self.theta), eg))
        alphafg = torch.trace(torch.mm(torch.mm(ef.t(), self.theta), eg))
        alphaff = torch.trace(torch.mm(torch.mm(ef.t(), self.theta), ef))
        eadapt = alphagg / (alphafg + alphagg) * eg + alphaff / (alphagf + alphaff) * ef
        madapt = torch.mm(eadapt, eadapt.t())  # 126*126
        adp = nn.functional.softmax(madapt, dim=1)
        if self.use_adaptive_mat_only:
            # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adj_mats = [adp]
        elif self.adaptive_mat_init is not None:
            # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # adaadjnumpy = adp.cpu().detach().numpy()
            # np.save("adaadjnumpy.npy", adaadjnumpy)
            # originadjnumpy = self.supports[0].cpu().detach().numpy()
            # np.save("originadjnumpy.npy", originadjnumpy)
            adj_mats = self.supports + [adp]

        else:
            adj_mats = self.supports
        for i in range(self.blocks * self.layers):
            residual = x
            x = torch.mul(torch.tanh(self.filter_convs[i](residual)), torch.sigmoid(self.gate_convs[i](residual)))

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if i == (self.blocks * self.layers - 2):  # -2 17
                break

            if self.use_graph_conv:
                x = x + self.graph_convs[i](x, adj_mats)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = self.output_layer(x)  # downsample to (batch_size, seq_len, n_vertex, features)
        attentionout = self.attention(x)
        y = torch.cat((attentionout, x), dim=3)
        out = self.finalFC(y)
        return out


if __name__ == '__main__':
    seqlen = 120
    nums = torch.ones((126, 126))
    temp = torch.randn((1, seqlen, 126, 17))
    geo_adj = np.random.randn(126, 126).astype(np.float32)
    flow_adj = np.random.randn(126, 126).astype(np.float32)
    linearx = AirspaceModel(layers=5, blocks=4, out_channels=seqlen, supports=[nums], geo_adj=geo_adj,
                            flow_adj=flow_adj)

    # layers=4,blocks=4 seqlen=60
    # layers=5,blocks=4 seqlen=125
    # layers=6,blocks=8 seqlen=505
    # x = F.softmax(temp,dim=2)
    print("input", temp.size(), linearx.receptive_field)
    out = linearx(temp)
    print(out.size())
    # k=torch.ones((126, 126)).unsqueeze(dim=0)
    # m = nn.AdaptiveAvgPool2d((7,7))
    # a,b=m(k).squeeze().shape
    # print(a,b)
    # k=torch.ones((1, 120, 126, 17))
    # for i in range(126):
    #     tmp=k[:,:,i,:]
    #     print(torch.flatten(tmp).size())
