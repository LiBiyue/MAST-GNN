import numpy as np
import torch
import torch.nn as nn


class self_attention(nn.Module):
    def __init__(self, dc, dk, dv):
        # dc是输入特征，
        # dk是中间特征，
        # dv是输出特征。
        super(self_attention, self).__init__()
        self.dk = dk
        self.softmax = nn.Softmax(dim=-1)  # 最后一个纬度做softmax
        self.WQ = nn.Linear(dc, dk, bias=False)
        self.WK = nn.Linear(dc, dk, bias=False)
        self.WV = nn.Linear(dc, dv, bias=False)
        nn.init.normal_(self.WQ.weight)
        nn.init.normal_(self.WK.weight)
        nn.init.normal_(self.WV.weight)
    def forward(self, H):
        """
        :param inputs: input features, [Batch,T,N,DC].
        :return:
            output features, [B,T,N,DV]
        """
        query = self.WQ(H)  # [B,T,N,DK]
        key = self.WK(H)  # [B,T,N,DK]
        value = self.WV(H)  # [B,T,N,DV]
        s = self.softmax(torch.einsum('onmk,kmio->oni', query, key.permute(*torch.arange(key.ndim - 1, -1, -1))) / np.sqrt(self.dk))  # [B,T,T]
        context = torch.einsum('onm,omki->onki', s, value)  # [B,T,N,DV]
        # print("query",query.size())
        # print("key",key.size())
        # print("value",value.size())
        # print("s",s.size())
        # print("context",context.size())
        return context


if __name__ == '__main__':
    tensor1 = torch.randn(5, 120, 126, 25)
    tensor2 = torch.randn(5, 120, 126, 25)
    # c = torch.einsum('onmk,kmio->oni', tensor1, tensor2.T)
    # print(c.shape)
    # d = torch.einsum('onm,omki->onki', c, tensor2)
    # print(d.shape)
    # tensor1 = torch.randn(120, 126, 25)
    # tensor2 = torch.randn(121, 126, 25)
    # c = torch.einsum('nmk,kmo->no', tensor1, tensor2.T)
    # print(c.shape)
    newtensor=torch.cat((tensor1,tensor2),dim=3)
    print(newtensor.size())
    tensor = torch.randn(5, 120, 126, 25)
    self_attentionmodel = self_attention(dc=25, dk=30, dv=40)
    tmp = self_attentionmodel(tensor)
    print(tmp.size())
