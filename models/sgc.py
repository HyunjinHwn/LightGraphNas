import torch
from torch import nn
from torch_geometric.nn import SGConv
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch import Tensor
from torch_geometric.typing import SparseTensor
from torch_sparse import SparseTensor as ST

from project.models.base import BaseGNN

class SGC(BaseGNN):
    """
    SGC: (Â)^K X W  ㅡ 비선형 없음, 드롭아웃만 선택적으로 사용
    - nlayers > 1는 의미가 없으므로 무시
    - args.K: 확산 횟수(K-hop), 기본 2~3 정도가 보통
    - edge_weight: 주면 그대로 사용(예: 정규화된 norm_adj), 안 주면 내부에서 GCN-style 정규화 수행
    """
    def __init__(self, nfeat, nhid, nclass, args):
        super().__init__(args)

        self.layers = nn.ModuleList([
            SGConv(
                in_channels=nfeat,
                out_channels=nclass,
                K=args.nlayers,
                cached=False,
                add_self_loops=True,
                bias=True,
                normalize=True,   # True-> D^{-1/2}(A+I)D^{-1/2} 
            )
        ])


    def forward_full(self, x, edge_index, EMB=False, edge_weight=None, **kwargs):
        x = self.layers[0](x, edge_index, edge_weight=edge_weight)
        if EMB:
            return [x]
        return x
    
    
    def forward_sampler(self, x_all, edge_index_batches, EMB=False, **kwargs):
        x = x_all
        conv = self.layers[0]

        for (edge_index, _, size) in edge_index_batches:
            n_src, n_dst = size
            device = x.device
            dtype = x.dtype

            # 1) (n_dst, n_src) 블록 (row=dst, col=src)
            src, dst = edge_index
            adj_t = ST.from_edge_index(
                torch.stack([dst, src], dim=0),
                sparse_sizes=(n_dst, n_src)
            )

            # 2) dst 대각 self-loop
            idx = torch.arange(n_dst, device=device)
            I = ST(row=idx, col=idx,
                value=torch.ones(n_dst, device=device, dtype=dtype),
                sparse_sizes=(n_dst, n_src))
            adj_t = adj_t + I

            # --- 여기서부터 수정 ---
            # coo로 꺼냈을 때 value가 None이면 암묵적 1로 채워주기
            row, col, val = adj_t.coo()
            if val is None:
                val = torch.ones(row.numel(), device=device, dtype=dtype)
            else:
                val = val.to(dtype)

            # bipartite GCN 정규화
            tmp = ST(row=row, col=col, value=val, sparse_sizes=(n_dst, n_src))
            d_dst = tmp.sum(dim=1).to(dtype)  # [n_dst]
            d_src = tmp.sum(dim=0).to(dtype)  # [n_src]
            inv_sqrt_d_dst = d_dst.clamp(min=1).pow(-0.5)
            inv_sqrt_d_src = d_src.clamp(min=1).pow(-0.5)

            val = inv_sqrt_d_dst[row] * val * inv_sqrt_d_src[col]
            adj_t = ST(row=row, col=col, value=val, sparse_sizes=(n_dst, n_src))
            # --- 수정 끝 ---

            x = adj_t.matmul(x[:n_src])

        x = conv.lin(x)
        return [x] if EMB else x
