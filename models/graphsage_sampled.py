import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import NeighborSampler

from project.models.base import BaseGNN

class GraphSageSample(BaseGNN):
    """
    GraphSAGE implementation compatible with the provided BaseGNN.

    - Supports full-graph and block (NeighborSampler) inference/training
    - Ignores edge_weight (standard SAGEConv does not use it)
    - self.normadj = False (GCN-style 정규화 사용하지 않음)
    - args.aggregator: 'mean' | 'max' | 'lstm' (default: 'mean')
    
    중요: sampling 추가, weighted 추가
    """

    def __init__(self, nfeat, nhid, nclass, args):
        super().__init__(args)
        self.normadj = False 

        # 1) sampling fanouts (예: [25, 10] 또는 args.sage_samples)
        # args에 없다면 기본값 사용
        self.sample_sizes = getattr(args, "sage_samples", [25] * self.nlayers)  
        assert len(self.sample_sizes) == self.nlayers, \
            f"sage_samples length must be {self.nlayers}"
            
        if self.nlayers == 1:
            self.layers.append(WeightedSAGEConv(in_channels=nfeat,
                                        out_channels=nclass,
                                        root_weight=True,   
                                        normalize=True     
                                        ))
        else:
            # input -> hidden
            self.layers.append(WeightedSAGEConv(in_channels=nfeat,
                                        out_channels=nhid,
                                        root_weight=True,   
                                        normalize=True  ))
            # hidden -> hidden (middle)
            for _ in range(self.nlayers - 2):
                self.layers.append(WeightedSAGEConv(in_channels=nhid,
                                            out_channels=nhid,
                                        root_weight=True,   
                                        normalize=True  ))
            # hidden -> output
            self.layers.append(WeightedSAGEConv(in_channels=nhid,
                                        out_channels=nclass,
                                        root_weight=True,   
                                        normalize=True  ))

    def forward_full(self, x, edge_index, edge_weight=None, EMB=False, **kwargs):
        """
        condensed graph에서도 full-batch가 아닌
        GraphSAGE-style neighbor sampling을 적용한 forward.
        """


        # 2) NeighborSampler 구성 (full graph에서 샘플링)
        sampler = NeighborSampler(
            edge_index,
            sizes=self.sample_sizes[::-1],     # shallow → deep 순서로 필요하므로 뒤집기
            batch_size=x.size(0),
            shuffle=False,
            return_e_id=True
        )

        # 3) 전체 노드를 하나의 배치로 넣어 호출 → blocks 얻기
        # loader는 (batch_size, n_id, adjs) 형태를 yield함
        (batch_size, n_id, adjs) = next(iter(sampler))  # 하나의 batch만

        # adjs = [(edge_index, e_id, size), ...]
        # shallow → deep 순서로 되어 있으므로 그냥 forward_sampler와 동일하게 처리
        # 단, x를 n_id 인덱스로 제한해야 함
        x_sampled = x[n_id]

        # 4) edge_weight도 adjs의 e_id를 이용해 slicing
        if edge_weight is not None:
            ew = edge_weight
        else:
            ew = None

        # 5) forward_sampler 스타일 실행
        emb_list = []
        out = F.dropout(x_sampled, p=self.dropout, training=self.training)

        for i, ((edge_index_b, e_id, size), conv) in enumerate(zip(adjs, self.layers)):
            if self.device is not None:
                edge_index_b = edge_index_b.to(self.device)
                e_id = e_id.to(self.device)
            # block weights
            ew_block = ew[e_id] if ew is not None else None

            # x for this block (x_src, x_dst)
            x_target = out[:size[1]]
            out = conv((out, x_target), 
                    edge_index_b, 
                    edge_weight=ew_block,
                    size=size)

            if EMB:
                emb_list.append(out)

            if i != self.nlayers - 1:
                out = self.activation(out)
                out = F.dropout(out, p=self.dropout, training=self.training)

        return emb_list if EMB else out


    # --- Block-wise propagation for NeighborSampler (edge_weight는 무시) ---
    def forward_sampler(self, x_all, edge_index_batches, edge_weight=None,EMB=False,  **kwargs):
        """
        edge_index_batches: iterable of blocks
          [(edge_index, e_id, size), ...]  from shallow -> deep
        size: (num_src, num_dst) for each block
        """
        x = x_all
        emb_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i, ((edge_index, _e_id, size), conv) in enumerate(zip(edge_index_batches, self.layers)):
            # x for this block: (x_src, x_dst)
            # 규약: 현재 블록의 타깃 노드들은 앞쪽에 모여 있어 x[:size[1]]
            x_target = x[:size[1]]
            x = conv((x, x_target), edge_index, size=size)
            if EMB: emb_list.append(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if EMB:
            return emb_list
        else:
            return x


import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 root_weight=True, normalize=True, bias=True):
        # mean 집계를 우리가 직접(가중 평균) 하므로 내부 aggr='add'
        super().__init__(aggr='add', node_dim=0)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_root  = nn.Linear(in_channels, out_channels, bias=False) if root_weight else None
        self.normalize = normalize
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: Tensor = None, size=None):
        # x_src, x_dst 분리 (샘플러 블록에서도 동작)
        if isinstance(x, Tensor):
            x_dst = x
        else:
            x, x_dst = x  # (x_src, x_dst)

        E = edge_index.size(1)
        
        if edge_weight is None:
            edge_weight = x.new_ones(E)

        # dst별 합으로 정규화 (가중 평균)
        dst = edge_index[1]
        deg = scatter_add(edge_weight, dst, dim=0, dim_size=x_dst.size(0))  # [N_dst]
        norm = edge_weight / (deg[dst].clamp_min(1e-12))

        # 이웃 집계: sum_j (norm_ij * x_j)
        out = self.propagate(edge_index, x=x, norm=norm, size=size)  # [N_dst, F]
        out = self.lin_neigh(out)

        if self.lin_root is not None:
            out = out + self.lin_root(x_dst)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j
