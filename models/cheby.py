import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from LightGraphNas.models.base import BaseGNN

class Cheby(BaseGNN):
    """
    ChebyNet implementation using ChebConv layers.
    """
    def __init__(self, nfeat, nhid, nclass, args):
        super().__init__(args)

        # Number of Chebyshev polynomial coefficients (K)
        self.normadj = False  # does normalization inside layer in checy way
        self.K = getattr(args, "K", 2)

        # ChebConv layers setup
        self.layers = nn.ModuleList([
            ChebConv(
                in_channels=nfeat,
                out_channels=nhid if self.nlayers > 1 else nclass,
                K=self.K,
                bias=True
            )
        ])

        if self.nlayers > 1:
            for _ in range(self.nlayers - 2):
                self.layers.append(ChebConv(
                    in_channels=nhid,
                    out_channels=nhid,
                    K=self.K,
                    bias=True
                ))
            self.layers.append(ChebConv(
                in_channels=nhid,
                out_channels=nclass,
                K=self.K,
                bias=True
            ))

    def forward_full(self, x, edge_index, EMB=False, edge_weight=None, **kwargs):
        emb_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight=edge_weight)
            if EMB: emb_list.append(x)
            if layer != self.layers[-1]:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.activation(x)  # Apply user-defined activation
        if EMB:
            return emb_list
        else:
            return x

    def forward_sampler(self, x_all, edge_index_batches, EMB=False, **kwargs):
        emb_list = []
        x = F.dropout(x_all, p=self.dropout, training=self.training)
        
        for (edge_index, e_id, size), conv in zip(edge_index_batches, self.layers):
            num_src, num_dst = size
            # 이 블록에 해당하는 특징만 남기기 (ChebConv는 bipartite/size 지원X)
            x_blk = x[:num_src]
            x_out = conv(x_blk, edge_index)
            x = x_out[:num_dst]

            if EMB: emb_list.append(x)
            if conv is not self.layers[-1]:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.activation(x)

        if EMB:
            return emb_list
        else:
            return x
