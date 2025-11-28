# from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv

from LightGraphNas.models.base import BaseGNN
from LightGraphNas.utils import *


class GCN(BaseGNN):
    def __init__(self, nfeat, nhid, nclass, args):
        super(GCN, self).__init__(args)
        if self.nlayers == 1:
            self.layers.append(GraphConv(nfeat, nclass, 
                                         add_self_loops=True, 
                                         normalize=True, 
                                         root_weight=False))
        else:
            self.layers.append(GraphConv(nfeat, nhid, 
                                         add_self_loops=True, 
                                         normalize=True, 
                                         root_weight=False))
            
            for i in range(self.nlayers - 2):
                self.layers.append(GraphConv(nhid, nhid, 
                                             add_self_loops=True, 
                                             normalize=True, 
                                             root_weight=False))
                
            self.layers.append(GraphConv(nhid, nclass,
                                         add_self_loops=True, 
                                         normalize=True, 
                                         root_weight=False))
