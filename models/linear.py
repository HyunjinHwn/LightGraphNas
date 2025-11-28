import torch
from torch import nn
import torch.nn.functional as F
from project.models.base import BaseGNN


class Linear(BaseGNN):
    """
    Simple Linear/MLP model (no graph convolution).
    - nlayers == 1:  X -> Linear(nfeat->nclass)
    - nlayers >= 2:  X -> [Linear + ReLU + Dropout] x (nlayers-1 with nhid) -> Linear -> nclass
    - 인접행렬(edge_index, edge_weight)은 무시
    """
    def __init__(self, nfeat, nhid, nclass, args):
        super().__init__(args)
        self.normadj = False  # 그래프 구조 사용 안 함

        self.layers = nn.ModuleList()

        if args.nlayers==1:
            self.layers.append(nn.Linear(nfeat, nclass))
        else:
            self.layers.append(nn.Linear(nfeat, nhid))
            for _ in range(args.nlayers - 2):
                self.layers.append(nn.Linear(nhid, nhid))
            self.layers.append(nn.Linear(nhid, nclass))

    def forward(self, x, edge_index=None, edge_weight=None, sampler=None, EMB=False, **kwargs):
        emb_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if EMB: emb_list.append(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if EMB:
            return emb_list
        else:
            return x
        

