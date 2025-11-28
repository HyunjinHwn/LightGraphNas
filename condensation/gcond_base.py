from collections import Counter

import torch.nn as nn

from LightGraphNas.coarsening import *
from LightGraphNas.condensation.utils import *
from LightGraphNas.models import *
from LightGraphNas.sparsification import *
from LightGraphNas.utils import *
from LightGraphNas.dataset.utils import save_reduced


class GCondBase:
    """
    A base class for graph condition generation and training.

    Parameters
    ----------
    setting : str
        The setting for the graph condensation process.
    data : object
        The data object containing the dataset.
    args : Namespace
        Arguments and hyperparameters for the model and training process.
    **kwargs : keyword arguments
        Additional arguments for initialization.
    """

    def __init__(self, setting, data, args, **kwargs):
        """
        Initializes a GCondBase instance.

        Parameters
        ----------
        setting : str
            The type of experimental setting.
        data : object
            The graph data object, which includes features, adjacency matrix, labels, etc.
        args : Namespace
            Arguments object containing hyperparameters for training and model.
        **kwargs : keyword arguments
            Additional optional parameters.
        """
        self.data = data
        self.args = args
        self.device = args.device
        self.setting = setting

        if args.method not in ['msgc']:
            self.labels_syn = self.data.labels_syn = self.generate_labels_syn(data)
            n = self.nnodes_syn = self.data.labels_syn.shape[0]
        else:
            n = self.nnodes_syn = int(data.feat_train.shape[0] * args.reduction_rate)
        self.d = d = data.feat_train.shape[1]
        print(f'target reduced size:{int(data.feat_train.shape[0] * args.reduction_rate)}')
        print(f'actual reduced size:{n}')


        self.feat_syn = nn.Parameter(torch.empty(n, d).to(self.device))
        if args.method not in ['sgdd', 'gcsntk', 'msgc']:
            self.pge = PGE(nfeat=d, nnodes=n, device=self.device, args=args).to(self.device)
            self.adj_syn = None

            # self.reset_parameters()
            self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
            self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
            print('adj_syn:', (n, n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        """
        Resets the parameters of the model.
        """
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
        self.pge.reset_parameters()

    def generate_labels_syn(self, data):
        """
        Generates synthetic labels to match the target number of samples.

        Parameters
        ----------
        data : object
            The graph data object, which includes features, adjacency matrix, labels, etc.

        Returns
        -------
        np.ndarray
            A numpy array of synthetic labels.
        """
        counter = Counter(data.labels_train.tolist())
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                # only clip labels with largest number of samples
                num_class_dict[c] = max(int(n * self.args.reduction_rate) - sum_, 1)
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
        self.data.num_class_dict = self.num_class_dict = num_class_dict
        if self.args.verbose:
            print(num_class_dict)
        return np.array(labels_syn)

    def init(self, with_adj=False, reuse_init=False):
        """
        Initializes synthetic features and (optionally) adjacency matrix.

        Parameters
        ----------
        with_adj : bool, optional
            Whether to initialize the adjacency matrix (default is False).

        Returns
        -------
        tuple
            A tuple containing the synthetic features and (optionally) the adjacency matrix.
        """
        args = self.args
        if args.init == 'clustering':
            if args.agg:
                agent = ClusterAgg(setting=args.setting, data=self.data, args=args)
            else:
                agent = Cluster(setting=args.setting, data=self.data, args=args)
        elif args.init == 'averaging':
            agent = Average(setting=args.setting, data=self.data, args=args)
        elif args.init == 'kcenter':
            agent = KCenter(setting=args.setting, data=self.data, args=args)
        elif args.init == 'herding':
            agent = Herding(setting=args.setting, data=self.data, args=args)
        elif args.init == 'cent_p':
            agent = CentP(setting=args.setting, data=self.data, args=args)
        elif args.init == 'cent_d':
            agent = CentD(setting=args.setting, data=self.data, args=args)
        else:
            agent = Random(setting=args.setting, data=self.data, args=args)
        
        temp = args.method
        args.method = args.init
        reduced_data = agent.reduce(self.data, verbose=True, save=True)
        args.method = temp
        if with_adj:
            return reduced_data.feat_syn, reduced_data.adj_syn
        else:
            return reduced_data.feat_syn


    def train_class_emb(self, model, labels_syn, args, original_graph=None):
        edge_index, features, labels = original_graph
        data = self.data
        feat_syn = self.feat_syn
        adj_syn = self.adj_syn
        edge_index_syn, edge_weight_syn = adj_to_edge_index_weight(adj_syn)
        loss = torch.tensor(0.0, device=self.device)
                
        # Loop over each class
        class_samplers =  data.retrieve_class_sampler(edge_index, args, shuffle=True)
        for c in range(data.nclass):
            for n_id, target, edge_index_batches in class_samplers[c]:
                if model.__class__.__name__ == 'ParallelWeightV1' or model.__class__.__name__ == 'ParallelWeightV2':
                    embs_real_list = model(features, edge_index, 
                                        edge_index_batches_tuple=(edge_index_batches, n_id, target), 
                                        sampler=True, EMB=True) # model list * class
                    embs_syn_list = model(feat_syn, edge_index_syn, edge_weight=edge_weight_syn, EMB=True) # model list * node * class
                    embs_syn_list = [embs_syn.mean(dim=0) for embs_syn in embs_syn_list] # model list * class
                    coeff = self.num_class_dict[c] / self.nnodes_syn
                    for embs_real, embs_syn in zip(embs_real_list, embs_syn_list):
                        loss += coeff * self.dist(embs_real, embs_syn, method=args.dis_metric)
                else:
                    if model.__class__.__name__ == 'APPNP' or model.__class__.__name__ == 'Linear':
                        # embeddings over class nodes 
                        embs_real = model(features, edge_index, EMB=True)
                        # mean embeddings of last layer
                        embs_mean_real = embs_real[-1][n_id[:target.shape[0]]].mean(dim=0)
                    elif model.__class__.__name__ == 'Parallel':
                        # mean last embeddings
                        embs_mean_real = model(features, edge_index, 
                                            edge_index_batches_tuple=(edge_index_batches, n_id, target), 
                                            sampler=True, EMB=True) # class
                    else:
                        # embeddings over class nodes 
                        embs_real = model(features[n_id], edge_index_batches, sampler=True, EMB=True)
                        # mean embeddings of last layer
                        embs_mean_real = embs_real[-1].mean(dim=0)

                    ######### syn forward랑 self.dist 차원
                    # embeddings over class nodes 
                    embs_syn = model(feat_syn, edge_index_syn, edge_weight=edge_weight_syn, EMB=True)
                    if type(embs_syn) == list:
                        embs_syn = embs_syn[-1] # only last layer # node * class
                    embs_mean_syn = embs_syn.mean(dim=0) # class
                    
                    # Compute matching loss between gradients
                    coeff = self.num_class_dict[c] / self.nnodes_syn
                    loss += coeff * self.dist(embs_mean_real, embs_mean_syn, method=args.dis_metric)
                    
        return loss
    

    def dist(self, x, y, method='l1'):
        """Distance objectives
        """
        if method == 'mse':
            dist_ = (x - y).pow(2).sum()
        elif method == 'l1':
            dist_ = (x - y).abs().sum()
        # elif method == 'l1_mean':
        #     n_b = x.shape[0]
        #     dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
        # elif method == 'cos':
        #     x = x.reshape(x.shape[0], -1)
        #     y = y.reshape(y.shape[0], -1)
        #     dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
        #                     (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))
        return dist_


    def train_class_grad(self, model, labels_syn, args, original_graph=None, soft=False):
        """
        Trains the model and computes the loss.

        Parameters
        ----------
        model : torch.nn.Module
            The model object.
        edge_index : torch.Tensor
            The edge_index .
        features : torch.Tensor
            The feature matrix.
        labels : torch.Tensor
            The actual labels.
        labels_syn : torch.Tensor
            The synthetic labels.
        args : Namespace
            Arguments object containing hyperparameters for training and model.

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        edge_index, features, labels = original_graph
        data = self.data
        feat_syn = self.feat_syn
        adj_syn = self.adj_syn
        edge_index_syn, edge_weight_syn = adj_to_edge_index_weight(adj_syn)
        loss = torch.tensor(0.0, device=self.device)

        if not soft:
            loss_fn = F.nll_loss
            # Convert labels to class indices if they are one-hot encoded
            if labels.dim() > 1:
                hard_labels = torch.argmax(labels, dim=-1)
            else:
                hard_labels = labels.long()
            if labels_syn.dim() > 1:
                hard_labels_syn = torch.argmax(labels_syn, dim=-1)
            else:
                hard_labels_syn = labels_syn.long()
        else:
            loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
            # Convert labels to one-hot encoding if they are class indices
            if labels.dim() == 1:
                hard_labels = labels
                soft_labels = F.one_hot(labels, num_classes=data.nclass).float()
            if labels_syn.dim() == 1:
                hard_labels_syn = labels
                soft_labels_syn = F.one_hot(labels_syn, num_classes=data.nclass).float()
            else:
                hard_labels_syn = torch.argmax(labels_syn, dim=-1)
                soft_labels_syn = labels_syn

        # Loop over each class
        class_samplers =  data.retrieve_class_sampler(edge_index, args, shuffle=True)
        for c in range(data.nclass):
            for n_id, target, edge_index_batches in class_samplers[c]:
                labels_real = (soft_labels if soft else hard_labels)[target].to(self.device)
                if model.__class__.__name__ == 'ParallelWeightV1' or model.__class__.__name__ == 'ParallelWeightV2':
                    output_real_list = model(features, edge_index, 
                                        edge_index_batches_tuple=(edge_index_batches, n_id, target), 
                                        sampler=True)
                    loss_real_list = []
                    for output_real in output_real_list:
                        loss_real_list.append( loss_fn(output_real, labels_real) )
                    gw_real_list = []
                    for model_idx, loss_real in enumerate(loss_real_list):
                        gw_real = torch.autograd.grad(loss_real, 
                                                      model.model_list[model_idx].parameters(), 
                                                      retain_graph=True)
                        gw_real_list += [g.detach().clone() for g in gw_real]
                    
                    output_syn_list = model(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
                    loss_syn_list = []
                    for output_syn in output_syn_list:
                        loss_syn_list.append(loss_fn(output_syn[hard_labels_syn == c], hard_labels_syn[hard_labels_syn == c]))
                    gw_syn_list = []
                    for model_idx, loss_syn in enumerate(loss_syn_list):
                        gw_syn = torch.autograd.grad(loss_syn, 
                                                     model.model_list[model_idx].parameters(), 
                                                     create_graph=True)
                        gw_syn_list += gw_syn
                    coeff = self.num_class_dict[c] / self.nnodes_syn
                    for gw_real, gw_syn in zip(gw_real_list, gw_syn_list):
                        ml = match_loss(gw_syn, gw_real, args, device=self.device)
                        loss += coeff * ml
                else:
                    if model.__class__.__name__ == 'APPNP' or model.__class__.__name__ == 'Linear':
                        output_real = model(features, edge_index)[n_id[:target.shape[0]]]
                    elif model.__class__.__name__ == 'Parallel':
                        output_real = model(features, edge_index, 
                                            edge_index_batches_tuple=(edge_index_batches, n_id, target), 
                                            sampler=True)
                    else:
                        output_real = model(features[n_id], edge_index_batches, sampler=True)
                        
                    loss_real = loss_fn(output_real, labels_real)
                    gw_real = torch.autograd.grad(loss_real, model.parameters(), retain_graph=True)
                    gw_real = [g.detach().clone() for g in gw_real]

                    output_syn = model(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
                    loss_syn = loss_fn(output_syn[hard_labels_syn == c], hard_labels_syn[hard_labels_syn == c])
                    # Compute gradients w.r.t. model parameters for synthetic data
                    gw_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

                    # Compute matching loss between gradients
                    coeff = self.num_class_dict[c] / self.nnodes_syn
                    ml = match_loss(gw_syn, gw_real, args, device=self.device)
                    loss += coeff * ml
                    
        return loss

    def get_loops(self, args):
        # Get the two hyper-parameters of outer-loop and inner-loop.
        # The following values are empirically good.
        """
        Retrieves the outer-loop and inner-loop hyperparameters.

        Parameters
        ----------
        args : Namespace
            Arguments object containing hyperparameters for training and model.

        Returns
        -------
        tuple
            Outer-loop and inner-loop hyperparameters.
        """
        return args.outer_loop, args.inner_loop


    def intermediate_evaluation(self, best_val, loss_avg=None, save=True):
        """
        Performs intermediate evaluation and saves the best model.

        Parameters
        ----------
        best_val : float
            The best validation accuracy observed so far.
        loss_avg : float
            The average loss.
        save : bool, optional
            Whether to save the model (default is True).

        Returns
        -------
        float
            The updated best validation accuracy.
        """
        data = self.data
        args = self.args
        if args.verbose:
            print('loss_avg: {}'.format(loss_avg))

        res = []

        args.logger.info(f'Testing with validation on {args.final_eval_model} model...')
        for i in range(args.run_inter_eval):
            res.append( 
                self.test_with_val(verbose=False, setting=args.setting, iters=args.eval_epochs))

        res = np.array(res).T
        current_val = res[0].mean()
        args.logger.warning('Val:  {:.4f} +/- {:.4f}'.format(100*current_val, 100*res[0].std()))
        args.logger.warning('Test: {:.4f} +/- {:.4f}'.format(100*res[1].mean(), 100*res[1].std()))

        if save and current_val > best_val:
            best_val = current_val
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)
        return best_val

    def test_with_val(self, verbose=False, setting='trans', iters=200, best_val=None):
        """
        Conducts validation testing and returns results.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose output (default is False).
        setting : str, optional
            The setting type (default is 'trans').
        iters : int, optional
            Number of iterations for validation testing (default is 200).

        Returns
        -------
        list
            A list containing validation results.
        """

        args, data, device = self.args, self.data, self.device
        
        model = eval(args.final_eval_model)(data.feat_syn.shape[1], args.hidden, data.nclass, args).to(device)

        acc_val = model.fit_with_val(data,
                                     train_iters=iters, normadj=True, verbose=False,
                                     setting=setting, reduced=True, best_val=best_val)

        model.eval()
        acc_test = model.test(data, setting=setting,verbose=False)
        # if verbose:
        #     print('Val Accuracy and Std:',
        #           repr([res.mean(0), res.std(0)]))
        return [acc_val, acc_test]
