import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor#, APPNP
from models.base import BaseGNN
from models.appnp import APPNP
from torch_geometric.nn import SGConv, GraphConv, GATConv,SAGEConv,ChebConv
from models.linear import Linear
from models.graphsage_sampled import WeightedSAGEConv

LAYER_MAP = {
    'Linear': Linear,
    'APPNP': APPNP,
    'SGC': SGConv,      
    'GCN': GraphConv,
    'GAT': GATConv,
    'GraphSage': SAGEConv,
    'GraphSageSample': WeightedSAGEConv,
    'Cheby': ChebConv,
}

# mapping activation 이름 → 실제 함수
ACT_MAP = {
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'relu': F.relu,
    'linear': lambda x: x,
    'softplus': F.softplus,
    'leakyrelu': F.leaky_relu,
    'relu6': F.relu6,
    'elu': F.elu,
}

class TwoLayerNet(BaseGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, args, mode='train',
                 layer1='GCN', layer2='Linear', activation='relu'):
        super().__init__(args)
        self.layer1 = layer1
        self.layer2 = layer2
        self.activation_name = activation
        if layer1 == 'GAT':
            hidden_dim_gat = hidden_dim // 4
            self.conv1 = GATConv(in_dim, hidden_dim_gat, heads=4, edge_dim=1, add_self_loops=False,)
        elif layer1 == 'APPNP':
            args.nlayers=1
            self.conv1 = APPNP(in_dim, hidden_dim, hidden_dim, args)
            args.nlayers=2
        elif layer1 == 'Cheby':
            self.conv1 = ChebConv(in_dim, hidden_dim, K=2)
        elif layer1 == 'Linear':
            self.conv1 = Linear(in_dim, hidden_dim, hidden_dim, args)
        else:
            self.conv1 = LAYER_MAP[layer1](in_dim, hidden_dim)
        if layer2 == 'GAT':
            self.conv2 = GATConv(hidden_dim, out_dim, heads=1, edge_dim=1, add_self_loops=False,)
        elif layer2 == 'APPNP':
            args.nlayers=1
            self.conv2 = APPNP(hidden_dim, hidden_dim, out_dim, args)
            args.nlayers=2
        elif layer2 == 'Cheby':
            self.conv2 = ChebConv(hidden_dim, out_dim, K=2)
        elif layer2 == 'Linear':
            self.conv2 = Linear(hidden_dim, hidden_dim, out_dim, args)
        else:
            self.conv2 = LAYER_MAP[layer2](hidden_dim, out_dim)
        self.activation = ACT_MAP[activation]

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.activation(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.activation(x)
        return x
    
    def parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    # def gatconv(self, x, adj):
    #     if isinstance(adj, SparseTensor):
    #         A = adj.coalesce()  # 안전하게 coalesce
    #         row, col, val = A.coo()          # val: [E], 0이 아닌 항만 저장됨
    #         edge_index = torch.stack([row, col], dim=0)  # [2, E]
    #         edge_attr = val.view(-1)                      # [E], 값은 0~1
    #     else:
    #         idx = (adj != 0).nonzero(as_tuple=False).t().contiguous()  # [2, E]
    #         edge_index = idx
    #         edge_attr = adj[idx[0], idx[1]].view(-1)                   # [E], 0~1
    #     return edge_index, edge_attr