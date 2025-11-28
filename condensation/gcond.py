from tqdm import trange

from project.condensation.gcond_base import GCondBase
from project.dataset.utils import save_reduced
from project.evaluation.utils import verbose_time_memory
from project.utils import *
from project.models import *


class GCond(GCondBase):
    """
    "Graph Condensation for Graph Neural Networks" https://cse.msu.edu/~jinwei2/files/GCond.pdf
    """
    def __init__(self, setting, data, args, **kwargs):
        super(GCond, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        pge = self.pge
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)
    
        # initialization the features
        edge_index, _ = adj_to_edge_index_weight(adj) # original graph doesn't have weight
        feat_init = self.init()
        self.feat_syn.data.copy_(feat_init)
        self.feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)


        outer_loop, inner_loop = self.get_loops(args)
        best_val = 0
        model = eval(args.condense_model)(self.d, args.hidden, data.nclass, args).to(self.device)
        for it in trange(args.epochs):
            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
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

                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                feat_syn_inner = self.feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                edge_index_syn, edge_weight_syn = adj_to_edge_index_weight(adj_syn_inner, device=self.device)
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner_list = model.forward(feat_syn_inner, edge_index_syn, edge_weight=edge_weight_syn, sampler=False)
                    if  type(output_syn_inner_list) != list:
                        output_syn_inner_list = [output_syn_inner_list]
                    loss_syn_inner = 0
                    for output_syn_inner in output_syn_inner_list:
                        loss_syn_inner += F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()

            loss_avg /= (data.nclass * outer_loop)

            if it in args.checkpoints:
                self.adj_syn = adj_syn_inner
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
