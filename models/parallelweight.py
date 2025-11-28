from copy import deepcopy

import torch.nn as nn

from  LightGraphNas.utils import *
from  LightGraphNas.models import *



class ParallelWeightV2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, args):
        super(ParallelWeightV2, self).__init__()
        self.model_fn_list = ['Linear', 'SGC','GCN','GAT','APPNP','Cheby','GraphSageSample']
        self.model_list = []
        for model_fn in self.model_fn_list:
            self.model_list.append(eval(model_fn)(nfeat, nhid, nclass, args))
        self.args = args
        
        
    def initialize(self):
        for model in self.model_list:
            model.initialize()


    def parameters(self):
        params = []
        for model in self.model_list:
            params += list(model.parameters())
        return params
    
    
    def named_parameters(self):
        params = []
        for model in self.model_list:
            params += list(model.named_parameters())
        return params
    
    
    def to(self, device):
        for model in self.model_list:
            model.to(device)
        self.device = device
        return self
    
    
    def forward(self, x, edge_index, edge_index_batches_tuple=None, edge_weight=None, sampler=False, EMB=False, **kwargs):
        if sampler:
            out = self.forward_sampler(x, edge_index, edge_index_batches_tuple, EMB=EMB, **kwargs)
        else:
            out = self.forward_full(x, edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs)
        if EMB:
            return out
        return [F.log_softmax(_out, dim=1) for _out in out]
    
    def forward_sampler(self, x, edge_index, edge_index_batches_tuple, EMB=False, **kwargs):
        edge_index_batches, n_id, target = edge_index_batches_tuple
        out_list = []
        for model in self.model_list:
            edge_index_batches_copy = deepcopy(edge_index_batches)
            if EMB:
                if model.__class__.__name__ == 'APPNP' or model.__class__.__name__ == 'Linear':
                    outs = model(x, edge_index, EMB=EMB)
                    outs = [emb[n_id[:target.shape[0]]].mean(dim=0) for emb in outs] # list(layer) of mean embedding
                else:
                    outs = model.forward_sampler(x[n_id], edge_index_batches=edge_index_batches_copy,
                                                    EMB=EMB, **kwargs) # node * class
                    outs = [emb.mean(dim=0) for emb in outs] # list(layer) of mean embedding
                out_list += outs[-1:] # model list * class
            else:
                if model.__class__.__name__ == 'APPNP' or model.__class__.__name__ == 'Linear':
                    outs = model(x, edge_index, EMB=EMB)
                    outs = outs[n_id[:target.shape[0]]]
                else:
                    outs = model.forward_sampler(x[n_id], edge_index_batches=edge_index_batches_copy, 
                                                    EMB=EMB, **kwargs)
                    outs = outs       
                out_list.append(outs)          
        return out_list # model list * class

    def forward_full(self, x, edge_index, edge_weight, EMB=False, **kwargs):
        out_list = []
        for model in self.model_list:
            outs = model.forward(x, edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs) # node * class
            if EMB:
                out_list += outs[-1:]
            else:
                out_list.append(outs)
        return out_list # model list * node * class
                
    
    def aggregate(self, out_list):
        if len(out_list)==1:
            out_tensor = torch.stack(out_list[0], dim=0)
            return torch.mean(out_tensor, dim=0)
        elif len(out_list) > 1:
            return out_list
        else:
            raise ValueError("out_list should not be empty")
        
    def fit_with_val(self, data, train_iters=600, verbose=False,
                     normadj=True, setting='trans', reduced=False, final_output=False, best_val=None, **kwargs):

        self.initialize()
        best_acc_val = []
        for model in self.model_list:
            model.train()
            best_acc_val.append(model.fit_with_val(data, train_iters=train_iters, verbose=verbose,
                                            normadj=normadj, setting=setting, reduced=reduced, 
                                            final_output=final_output, best_val=best_val, **kwargs))
        return self.aggregate(best_acc_val)

    
    @torch.no_grad()
    def test(self, data, setting='trans', verbose=False):
        acc_test = []
        for model in self.model_list:
            model.eval()
            acc_test.append(model.test(data, setting=setting, verbose=verbose))
        return self.aggregate(acc_test)

    @torch.no_grad()
    def predict(self, features=None, adj=None, normadj=True, output_layer_features=False):
        predicted = []
        for model in self.model_list:
            model.eval()
            predicted.append(model.predict(features, adj, normadj=normadj, output_layer_features=output_layer_features))
        return self.aggregate(predicted)
    
class ParallelWeightV1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, args):
        super(ParallelWeightV1, self).__init__()
        self.model_fn_list = ['Linear', 'SGC','GCN','GAT','APPNP','Cheby','GraphSage']
        self.model_list = []
        for model_fn in self.model_fn_list:
            self.model_list.append(eval(model_fn)(nfeat, nhid, nclass, args))
        self.args = args
        # self.weights = nn.Parameter(torch.ones(len(self.model_list))/len(self.model_list))
        
        
    def initialize(self):
        for model in self.model_list:
            model.initialize()


    def parameters(self):
        params = []
        for model in self.model_list:
            params += list(model.parameters())
        return params
    
    
    def named_parameters(self):
        params = []
        for model in self.model_list:
            params += list(model.named_parameters())
        return params
    
    
    def to(self, device):
        for model in self.model_list:
            model.to(device)
        self.device = device
        return self
    
    
    def forward(self, x, edge_index, edge_index_batches_tuple=None, edge_weight=None, sampler=False, EMB=False, **kwargs):
        if sampler:
            out = self.forward_sampler(x, edge_index, edge_index_batches_tuple, EMB=EMB, **kwargs)
        else:
            out = self.forward_full(x, edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs)
        if EMB:
            return out
        return [F.log_softmax(_out, dim=1) for _out in out]
    
    def forward_sampler(self, x, edge_index, edge_index_batches_tuple, EMB=False, **kwargs):
        edge_index_batches, n_id, target = edge_index_batches_tuple
        out_list = []
        for model in self.model_list:
            edge_index_batches_copy = deepcopy(edge_index_batches)
            if EMB:
                if model.__class__.__name__ == 'APPNP' or model.__class__.__name__ == 'Linear':
                    outs = model(x, edge_index, EMB=EMB)
                    outs = [emb[n_id[:target.shape[0]]].mean(dim=0) for emb in outs] # list(layer) of mean embedding
                else:
                    outs = model.forward_sampler(x[n_id], edge_index_batches=edge_index_batches_copy,
                                                    EMB=EMB, **kwargs) # node * class
                    outs = [emb.mean(dim=0) for emb in outs] # list(layer) of mean embedding
                out_list += outs[-1:] # model list * class
            else:
                if model.__class__.__name__ == 'APPNP' or model.__class__.__name__ == 'Linear':
                    outs = model(x, edge_index, EMB=EMB)
                    outs = outs[n_id[:target.shape[0]]]
                else:
                    outs = model.forward_sampler(x[n_id], edge_index_batches=edge_index_batches_copy, 
                                                    EMB=EMB, **kwargs)
                    outs = outs       
                out_list.append(outs)          
        return out_list # model list * class

    def forward_full(self, x, edge_index, edge_weight, EMB=False, **kwargs):
        out_list = []
        for model in self.model_list:
            outs = model.forward(x, edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs) # node * class
            if EMB:
                out_list += outs[-1:]
            else:
                out_list.append(outs)
        return out_list # model list * node * class
                
    
    def aggregate(self, out_list):
        if len(out_list)==1:
            out_tensor = torch.stack(out_list[0], dim=0)
            return torch.mean(out_tensor, dim=0)
        elif len(out_list) > 1:
            return out_list
        else:
            raise ValueError("out_list should not be empty")
        
    def fit_with_val(self, data, train_iters=600, verbose=False,
                     normadj=True, setting='trans', reduced=False, final_output=False, best_val=None, **kwargs):

        self.initialize()
        best_acc_val = []
        for model in self.model_list:
            model.train()
            best_acc_val.append(model.fit_with_val(data, train_iters=train_iters, verbose=verbose,
                                            normadj=normadj, setting=setting, reduced=reduced, 
                                            final_output=final_output, best_val=best_val, **kwargs))
        return self.aggregate(best_acc_val)

    
    @torch.no_grad()
    def test(self, data, setting='trans', verbose=False):
        acc_test = []
        for model in self.model_list:
            model.eval()
            acc_test.append(model.test(data, setting=setting, verbose=verbose))
        return self.aggregate(acc_test)

    @torch.no_grad()
    def predict(self, features=None, adj=None, normadj=True, output_layer_features=False):
        predicted = []
        for model in self.model_list:
            model.eval()
            predicted.append(model.predict(features, adj, normadj=normadj, output_layer_features=output_layer_features))
        return self.aggregate(predicted)
    