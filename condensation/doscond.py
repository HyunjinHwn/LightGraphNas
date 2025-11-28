from tqdm import trange

from LightGraphNas.condensation.gcond_base import GCondBase
from LightGraphNas.evaluation.utils import verbose_time_memory
from LightGraphNas.utils import *
from LightGraphNas.models import *


class DosCond(GCondBase):
    """
    "Condensing Graphs via One-Step Gradient Matching" https://arxiv.org/abs/2206.07746
    """
    def __init__(self, setting, data, args, **kwargs):
        super(DosCond, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):

        args = self.args
        pge = self.pge
        feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        # initialization the features
        edge_index, _ = adj_to_edge_index_weight(adj) # original graph doesn't have weight
        feat_init = self.init()
        self.feat_syn.data.copy_(feat_init)
        adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))

        outer_loop, inner_loop = self.get_loops(args)
        best_val = 0

        # seed_everything(args.seed + it)
        model = eval(args.condense_model)(feat_syn.shape[1], args.hidden, data.nclass, args).to(self.device)
        for it in trange(args.epochs):
            model.initialize()
            model.train()
            loss_avg = 0

            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                self.adj_syn = normalize_adj_tensor(adj_syn, sparse=False)
                loss = self.train_class_grad(model, labels_syn, args, original_graph=(edge_index, features, labels))
                loss_avg += loss.item()
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                self.optimizer_pge.step()
                self.optimizer_feat.step()

            loss_avg /= (data.nclass * outer_loop)

            if it in args.checkpoints:
                self.adj_syn = pge.inference(self.feat_syn)
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
