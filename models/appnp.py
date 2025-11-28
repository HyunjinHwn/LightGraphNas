import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as APPNPLayer

from LightGraphNas.models.base import BaseGNN


class APPNP(BaseGNN):
    """
    APPNP: f_θ(X) (MLP로 로짓 생성) -> APPNP 전파(K, alpha)
    - self.normadj = True  (주어진 코드베이스 약속: GCN/APPNP는 정규화 인접행렬 사용)
    - nlayers: MLP의 층 수 (1이면 Linear(nfeat->nclass)만 적용)
    - args.K: 전파 스텝 수 (기본 10~20 자주 사용)
    - args.alpha: teleport 확률 (기본 0.1)
    - 드롭아웃은 MLP 사이와 APPNP 전파(dropout 인자) 모두에 적용
    - sampler 모드는 지원하지 않음(Full-batch 전파 필요)
    """
    def __init__(self, nfeat, nhid, nclass, args):
        super().__init__(args)

        self.K = getattr(args, "K", 10)
        self.alpha = getattr(args, "alpha", 0.1)
        # 전파 과정에서 사용할 dropout(별도 지정 없으면 모델의 dropout 사용)
        self.propagation_dropout = getattr(args, "prop_dropout", self.dropout)

        # --- MLP 구성 ---
        # BaseGNN의 self.activation을 그대로 사용 (args.activation)
        self.mlp_layers = nn.ModuleList()
        if self.nlayers == 1:
            self.mlp_layers.append(nn.Linear(nfeat, nclass))
        else:
            self.mlp_layers.append(nn.Linear(nfeat, nhid))
            for _ in range(self.nlayers - 2):
                self.mlp_layers.append(nn.Linear(nhid, nhid))
            self.mlp_layers.append(nn.Linear(nhid, nclass))

        # --- APPNP 전파 레이어 ---
        # torch_geometric.nn.APPNP(K, alpha, dropout) : dropout은 전파 과정에서의 feature dropout
        self.propagation = APPNPLayer(K=self.K, alpha=self.alpha, dropout=self.propagation_dropout, 
                                      cached = False, add_self_loops = True, normalize = True)

    # BaseGNN.initialize()는 self.layers만 reset하므로, MLP도 리셋되도록 오버라이드
    def initialize(self):
        for lin in self.mlp_layers:
            if hasattr(lin, "reset_parameters"):
                lin.reset_parameters()
        if hasattr(self.propagation, "reset_parameters"):
            self.propagation.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, sampler=False, EMB=False, **kwargs):
        if sampler:
            out = self.forward_sampler(x, edge_index_batches=edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs)
            return 'error'
        else:
            out =  self.forward_full(x, edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs)
            if EMB:
                return out
        return F.log_softmax(out, dim=1)
    
    # full-batch 전파
    def forward_full(self, x, edge_index, edge_weight=None, EMB=False, **kwargs):
        # 1) MLP로 로짓 생성
        emb_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, lin in enumerate(self.mlp_layers):
            x = lin(x)
            if EMB:
                emb_list.append(x)
            if i != len(self.mlp_layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # 2) APPNP 전파
        x = self.propagation(x, edge_index, edge_weight=edge_weight)
        if EMB:
            emb_list.append(x)
            return emb_list
        return x

    # NeighborSampler 기반 샘플링 전파는 APPNP 특성상 지원하지 않음(전체 그래프 필요)
    def forward_sampler(self, x_all, edge_index_batches, EMB=False,  **kwargs):
        raise NotImplementedError("APPNP는 샘플러 모드를 지원하지 않습니다. full-batch로 사용해주세요.")
