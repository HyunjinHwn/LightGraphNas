from tqdm import trange

from LightGraphNas.condensation.gcond_base import GCondBase
from LightGraphNas.dataset.utils import save_reduced
from LightGraphNas.evaluation.utils import verbose_time_memory
from LightGraphNas.utils import *
from LightGraphNas.models import *
import torch.nn.functional as F


class GCDM(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(GCDM, self).__init__(setting, data, args, **kwargs)
    @verbose_time_memory
    def reduce(self, data, verbose=True):
        """
        GCDM reduce:
        - K1회: min over synthetic graph vars (feat_syn X', pge g_θ) using class-centroid distance loss
        - K2회: max over model params θ on the SAME loss (worst-case discriminator)
        Notes:
        * outer(min): model params are FROZEN (no grad to θ)
        * inner(max): synthetic graph is DETACHED (no grad to X', g_θ)
        """
        args = self.args
        pge = self.pge

        # --- load original graph tensors ---
        if args.setting == 'trans':
            features, adj, labels = to_tensor(
                data.feat_full, data.adj_full, label=data.labels_full, device=self.device
            )
        else:
            features, adj, labels = to_tensor(
                data.feat_train, data.adj_train, label=data.labels_train, device=self.device
            )

        edge_index, _ = adj_to_edge_index_weight(adj)  # original graph assumed unweighted
        # --- init synthetic features/labels ---
        feat_init = self.init()
        self.feat_syn.data.copy_(feat_init)
        self.feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)

        # --- loop sizes ---
        outer_loop, inner_loop = self.get_loops(args)
        best_val = 0.0

        # --- build condense-time model (discriminator Φ_θ) ---
        model = eval(args.condense_model)(self.d, args.hidden, data.nclass, args).to(self.device)

        for it in trange(args.epochs):
            # (re)initialize model params each epoch (as in your original code)
            model.initialize()
            model.train()

            # ===== K1: OUTER MIN — update X', g_θ to REDUCE class-centroid distance =====
            # freeze model params (no grad to θ in outer stage)
            for p in model.parameters():
                p.requires_grad_(False)

            loss_avg = 0.0
            adj_syn_inner = None  # will be set below

            for ol in range(outer_loop):
                # build/normalize synthetic adjacency from current X'
                adj_syn_raw = pge(self.feat_syn)                          # A' = g_θ(X')
                self.adj_syn = normalize_adj_tensor(adj_syn_raw, sparse=False)
                
                # class-centroid distance loss (same in outer & inner)
                loss_min = self.train_class_emb(
                    model, labels_syn, args,
                    original_graph=(edge_index, features, labels)
                )
                loss_avg += loss_min.item()

                # update X' and/or g_θ only
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss_min.backward()

                # your update schedule: sometimes update g_θ, sometimes X'
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

            # Snapshot a detached synthetic graph for the inner MAX stage
            with torch.no_grad():
                feat_syn_inner = self.feat_syn.detach()
                adj_syn_inner = normalize_adj_tensor(pge.inference(feat_syn_inner), sparse=False)

            # ===== K2: INNER MAX — update θ to INCREASE the same centroid-distance loss =====
            # unfreeze model params (grad to θ), and keep synthetic graph detached
            for p in model.parameters():
                p.requires_grad_(True)

            # Use standard Adam; we'll do ascent by backprop on (-loss)
            model_parameters = list(model.parameters())
            self.optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            
            # temporarily swap handles so train_class_emb reads the DETACHED synthetic graph
            feat_bak, adj_bak = self.feat_syn, self.adj_syn
            self.feat_syn, self.adj_syn = feat_syn_inner, adj_syn_inner

            for _ in range(inner_loop):
                self.optimizer_model.zero_grad()
                loss_adv = self.train_class_emb(
                    model, labels_syn, args,
                    original_graph=(edge_index, features, labels)
                )
                # gradient ASCENT on model params:
                (-loss_adv).backward()
                self.optimizer_model.step()

            # restore handles for next iteration
            self.feat_syn, self.adj_syn = feat_bak, adj_bak

            # ===== bookkeeping / checkpointing =====
            loss_avg /= max(1, (data.nclass * outer_loop))

            if it in args.checkpoints:
                # write back the latest (detached) synthetic graph snapshot
                # (adj_syn_inner is normalized; feat_syn already lives on device)
                self.adj_syn = adj_syn_inner
                data.adj_syn  = self.adj_syn.detach()
                data.feat_syn = self.feat_syn.detach()
                data.labels_syn = labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
