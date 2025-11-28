import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from project.models.base import BaseGNN
from project.utils import *

class GAT(BaseGNN):
    """
    GAT:
    - normadj = False (GCN 스타일 정규화 미사용; 학습으로 attention 학습)
    - 중간 레이어: concat=True, heads=args.heads
    - 마지막 레이어: concat=False, heads=args.out_heads(기본 1) -> 출력 차원 = nclass
    - feature dropout은 BaseGNN에서 처리, 여기서는 GATConv의 attention dropout만 파라미터로 전달
    - sampler 모드: BaseGNN.forward_sampler가 bipartite 입력((x_src, x_dst), size)을 넘기므로 그대로 호환
    """
    def __init__(self, nfeat, nhid, nclass, args):
        super(GAT, self).__init__(args)
        self.normadj = False  # GAT은 인접 정규화 불필요

        heads = getattr(args, "heads", 4)              # 중간 레이어 멀티헤드 개수
        out_heads = getattr(args, "out_heads", 1)      # 마지막 레이어 헤드 수(보통 1)
        attn_dropout = self.dropout
        negative_slope = getattr(args, "alpha", 0.2)   # LeakyReLU 음의 기울기
        # args에서 가져오되, 없으면 default
        self.ratio = getattr(args, "gat_prune_ratio", 0.2)      # 각 노드 degree의 몇 %를 살릴지
        self.min_k = getattr(args, "gat_prune_min_k", 1)        # 최소 이웃 수
        self.max_k = getattr(args, "gat_prune_max_k", 64)       # 최대 이웃 수 (None이면 제한 없음)

        self.layers = nn.ModuleList()

        if self.nlayers == 1:
            # 단일 레이어: 바로 nclass로 투사 (concat=False)
            self.layers.append(
                GATConv(
                    in_channels=nfeat,
                    out_channels=nclass,
                    heads=out_heads,
                    concat=False,
                    dropout=attn_dropout,
                    negative_slope=negative_slope,
                    add_self_loops=True,
                    bias=True,
                )
            )
        else:
            # 첫 레이어
            # 중간 레이어의 출력 총 차원 = out_channels * heads
            # 사용자가 준 nhid가 heads로 나누어 떨어지지 않으면, 가장 가까운 배수로 맞춥니다.
            out_per_head = max(1, math.ceil(nhid / heads))
            hidden_total = out_per_head * heads

            self.layers.append(
                GATConv(
                    in_channels=nfeat,
                    out_channels=out_per_head,
                    heads=heads,
                    concat=True,                 # 총 차원 = out_per_head * heads
                    dropout=attn_dropout,
                    negative_slope=negative_slope,
                    add_self_loops=True,
                    bias=True,
                )
            )

            # (중간) 2..(L-1) 레이어
            for _ in range(self.nlayers - 2):
                # 이전 총 출력 차원 = hidden_total
                self.layers.append(
                    GATConv(
                        in_channels=hidden_total,
                        out_channels=out_per_head,
                        heads=heads,
                        concat=True,
                        dropout=attn_dropout,
                        negative_slope=negative_slope,
                        add_self_loops=True,
                        bias=True,
                    )
                )

            # 마지막 레이어: concat=False로 로짓 차원을 nclass로 맞춤
            self.layers.append(
                GATConv(
                    in_channels=hidden_total,
                    out_channels=nclass,
                    heads=out_heads,
                    concat=False,               # 최종 출력 = nclass (out_heads가 1이 아니어도 concat False면 out_channels 유지)
                    dropout=attn_dropout,
                    negative_slope=negative_slope,
                    add_self_loops=True,
                    bias=True,
                )
            )
    

    # def forward_full(self, x, edge_index, edge_weight=None, EMB=False, **kwargs):
    #     emb_list = []
    #     x = F.dropout(x, p=self.dropout, training=self.training)
        
    #     for i, layer in enumerate(self.layers):
    #         # GATConv는 edge_attr를 사용 (가중치가 없으면 None 전달)
    #         x = layer(x, edge_index, edge_attr=edge_weight)
    #         if EMB: emb_list.append(x)
    #         if i != self.nlayers - 1:
    #             x = F.dropout(x, p=self.dropout, training=self.training)
    #             x = self.activation(x)
                
    #     if EMB:
    #         return emb_list
    #     else:
    #         return x
    
    def forward(self, x, edge_index, edge_weight=None, sampler=False, EMB=False, **kwargs):
        if sampler:
            out = self.forward_sampler(x, edge_index_batches=edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs)
        else:
            out =  self.forward_full(x, edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs)
        if EMB:
            return out
        return F.log_softmax(out, dim=1)
    
    def forward_full(
        self,
        x,
        edge_index,
        edge_weight=None,
        EMB=False,
        use_pruning: bool = False,
        **kwargs,
    ):
        """
        condensation 단계:
            -> use_pruning=False  (기본값)  => 기존과 동일하게 full dense GAT

        evaluation / condensed 학습 단계:
            -> use_pruning=True   => node별 top-k pruning + weighted edge_attr
        """
        emb_list = []

        # --- pruning 설정 (adaptive k) ---
        if use_pruning:

            num_nodes = x.size(0)
            edge_index_used, edge_weight_used = self.nodewise_topk_prune(
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=num_nodes,
                ratio=self.ratio,
                min_k=self.min_k,
                max_k=self.max_k,
            )
        else:
            edge_index_used, edge_weight_used = edge_index, edge_weight

        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            # GATConv는 edge_attr에 weight를 받음 (None이면 unweighted attention)
            x = layer(x, edge_index_used, edge_attr=edge_weight_used)
            if EMB:
                emb_list.append(x)
            if i != self.nlayers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.activation(x)

        if EMB:
            return emb_list
        else:
            return x

    def forward_sampler(self, x_all, edge_index_batches, EMB=False, edge_weight=None, **kwargs):
        """
        edge_index_batches: [(edge_index, e_id, size), ...]
        BaseGNN는 (x_src, x_dst) 형태와 size를 conv에 전달하므로 동일하게 맞춤.
        """
        x = x_all
        emb_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for (edge_index, e_id, size), conv in zip(edge_index_batches, self.layers):
            x_target = x[:size[1]]
            ew_blk = None if edge_weight is None else edge_weight[e_id]
            # GATConv는 bipartite 입력을 ((x_src, x_dst), edge_index, edge_attr, size)로 받음
            x = conv((x, x_target), edge_index, edge_attr=ew_blk, size=size)
            if EMB: emb_list.append(x)
            if conv is not self.layers[-1]:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.activation(x)
                
        if EMB:
            return emb_list
        else:
            return x
    
    import torch

    def nodewise_topk_prune(self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int,
        ratio: float = 0.2,
        min_k: int = 1,
        max_k: Optional[int] = None
    ):
        """
        각 dst 노드별로 edge_weight 상위 k개만 남기는 pruning.
        k_i = clamp(int(deg_i * ratio), min_k, max_k)

        edge_index: [2, E]
        edge_weight: [E] or None (None이면 모두 1로 간주)
        """
        device = edge_index.device
        E = edge_index.size(1)
        dst = edge_index[1]

        if edge_weight is None:
            ew = torch.ones(E, device=device)
        else:
            ew = edge_weight

        keep_mask = torch.zeros(E, dtype=torch.bool, device=device)

        for v in range(num_nodes):
            idx_v = (dst == v).nonzero(as_tuple=False).view(-1)
            deg_v = idx_v.numel()
            if deg_v == 0:
                continue

            k = int(deg_v * ratio)
            if k < min_k:
                k = min_k
            if max_k is not None:
                k = min(k, max_k)
            k = min(k, deg_v)  # deg보다 크게는 못 뽑음

            # v 노드로 들어오는 edge 중 weight 상위 k개 선택
            _, top_local = torch.topk(ew[idx_v], k, largest=True, sorted=False)
            keep_mask[idx_v[top_local]] = True

        pruned_edge_index = edge_index[:, keep_mask]
        pruned_edge_weight = ew[keep_mask]

        return pruned_edge_index, pruned_edge_weight
