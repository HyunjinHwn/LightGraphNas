from copy import deepcopy

import torch.nn as nn
import torch.optim as optim

from project.utils import *

class BaseGNN(nn.Module):
    
    def __init__(self, args):

        super(BaseGNN, self).__init__()
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.dropout = args.cond_dropout
        self.cond_dropout = args.cond_dropout
        self.eval_dropout = args.eval_dropout
        self.device = args.device
        self.layers = nn.ModuleList([])
        self.nlayers = args.nlayers
        self.metric = args.metric
        self.normadj = None # Explicitly defined 
                            # (automatic self-loop in layer) APPNP GCN 
                            # False: GAT, SAGE, SGC, Cheb
        self.selfloop = None # Explicitly defined
                            # (automatic self-loop in layer) APPNP GCN 
                            # False: Cheb
                            
        activation_functions = {
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'relu': F.relu,
            'linear': lambda x: x,
            'softplus': F.softplus,
            'leakyrelu': F.leaky_relu,
            'relu6': F.relu6,
            'elu': F.elu
        }
        self.activation = activation_functions.get(args.activation)
        self.loss = F.nll_loss
        
    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    
    def forward_full(self, x, edge_index, edge_weight, EMB=False, **kwargs):
        emb_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=edge_weight)
            if EMB: emb_list.append(x)
            if i != self.nlayers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.activation(x)
        if EMB:
            return emb_list
        else:
            return x
    
    def forward_sampler(self, x_all, edge_index_batches, EMB=False, **kwargs):
        # adjs: [(edge_index, e_id, size), ...]  (얕은 레이어 -> 깊은 레이어 순)
        x = x_all  
        emb_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for (edge_index, e_id, size), conv in zip(edge_index_batches, self.layers):  # size = (num_src, num_dst)
            x_target = x[:size[1]]      # 이 블록의 타겟 노드들
            # 블록 전파: x를 (src, dst)로 나눠서 전달 + size 지정
            x = conv((x, x_target), edge_index, size=size)
            if EMB: emb_list.append(x)
            if conv is not self.layers[-1]:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.activation(x)
        if EMB:
            return emb_list
        else:
            return x


    def forward(self, x, edge_index, edge_weight=None, sampler=False, EMB=False, **kwargs):
        if sampler:
            out = self.forward_sampler(x, edge_index_batches=edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs)
        else:
            out =  self.forward_full(x, edge_index, edge_weight=edge_weight, EMB=EMB, **kwargs)
        if EMB:
            return out
        return F.log_softmax(out, dim=1)
    
    
    def fit_with_val(self, data, train_iters=600, verbose=False, 
                     setting='trans', reduced=False, **kwargs):
        self.dropout = self.eval_dropout
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], 'dropout'):
                self.layers[i].dropout = self.eval_dropout
                                             
        self.initialize()
        
        # data for training
        if reduced:
            adj, features, labels, labels_val = to_tensor(data.adj_syn, data.feat_syn, data.labels_syn,
                                                          label2=data.labels_val,
                                                          device=self.device)

        elif setting == 'trans':
            adj, features, labels, labels_val = to_tensor(data.adj_full, data.feat_full, label=data.labels_train,
                                                          label2=data.labels_val, device=self.device)
        else:
            adj, features, labels, labels_val = to_tensor(data.adj_train, data.feat_train, label=data.labels_train,
                                                          label2=data.labels_val, device=self.device)
        
        if setting == 'ind': 
            feat_full, adj_full = data.feat_val, data.adj_val
        else: 
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = to_tensor(feat_full, adj_full, device=self.device)
        
        
        if self.__class__.__name__ == 'GAT':
            if not is_sparse_tensor(adj):
                adj = adj.to_sparse()
            if not is_sparse_tensor(adj_full):
                adj_full = adj_full.to_sparse()
        
        edge_index, edge_weight = adj_to_edge_index_weight(adj, device=self.device)
        edge_index_full, edge_weight_full = adj_to_edge_index_weight(adj_full, device=self.device)

        labels = to_tensor(label=labels, device=self.device)

        best_acc_val = 0
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.train()
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(features, edge_index, edge_weight=edge_weight, use_pruning=True)
            loss_train = self.loss(output if output.shape[0] == labels.shape[0] else output[data.idx_train], labels)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                acc_train = accuracy(output if output.shape[0] == labels.shape[0] else output[data.idx_train], labels)
                print('Epoch {}, training acc: {}'.format(i, acc_train))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, edge_index_full, edge_weight=edge_weight_full)
                acc_val = self.metric(output if output.shape[0] == labels_val.shape[0] else output[data.idx_val],
                                   labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    weights = deepcopy(self.state_dict())
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        self.dropout = self.cond_dropout
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], 'dropout'):
                self.layers[i].dropout = self.cond_dropout
        return best_acc_val.item()

    
    @torch.no_grad()
    def predict(self, features=None, adj=None):

        self.eval()
        features, adj = to_tensor(features, adj, device=self.device)
        edge_index, edge_weight = adj_to_edge_index_weight(adj, device=self.device)

        return self.forward(features, edge_index, edge_weight=edge_weight)

    @torch.no_grad()
    def test(self, data, setting, verbose):
        labels_test = to_tensor(label=data.labels_test, device=self.device)

        if setting == 'ind':
            output = self.predict(data.feat_test, data.adj_test)
            loss_test = self.loss(output, labels_test)
            acc_test = self.metric(output, labels_test).item()

            if verbose:
                print("Test set results:",
                      f"loss= {loss_test.item():.4f}",
                      f"accuracy= {acc_test:.4f}")
        else:
            output = self.predict(data.feat_full, data.adj_full)
            loss_test = self.loss(output[data.idx_test], labels_test)
            acc_test = self.metric(output[data.idx_test], labels_test).item()
            if verbose:
                print("Test full set results:", f"loss= {loss_test.item():.4f}", f"accuracy= {acc_test:.4f}")
        return acc_test