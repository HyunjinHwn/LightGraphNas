import os
import sys
from LightGraphNas.dataset.convertor import *
from LightGraphNas.utils import is_sparse_tensor



def sparsify(model_type, adj_syn, args, verbose=False):
    """
    Applies sparsification to the adjacency matrix based on the model type and given arguments.

    This function modifies the adjacency matrix to make it sparser according to the model type and method specified.
    For specific methods and datasets, it adjusts the threshold used for sparsification.

    Parametersm
    ----------
    model_type : str
        The type of model used, which determines the sparsification strategy. Can be 'MLP', 'GAT', or other.
    adj_syn : torch.Tensor
        The adjacency matrix to be sparsified.
    args : argparse.Namespace
        Command-line arguments and configuration parameters which may include method-specific settings.
    verbose : bool, optional
        If True, prints information about the sparsity of the adjacency matrix before and after sparsification.
        Default is False.

    Returns
    -------
    adj_syn : torch.Tensor
        The sparsified adjacency matrix.
    """
    threshold = 0
    if args.method in ['gcond', 'doscond']:
        threshold = args.threshold
    if args.method in ['random', 'kcenter', 'herding', 'cent_p', 'cent_d', 'simgc','msgc']:
        threshold = 0.0
    if model_type == 'GAT':
        if args.method in ['gcond', 'doscond']:
            if args.dataset in ['cora', 'citeseer']:
                threshold = 0.5  # Make the graph sparser as GAT does not work well on dense graph
            else:
                threshold = 0.1
    # print('Sparsity:', adj_syn.nonzero().shape[0] / adj_syn.numel())
    # if threshold > 0:
    #     adj_syn[adj_syn < threshold] = 0
    #     print('Sparsity after truncating:', adj_syn.nonzero().shape[0] / adj_syn.numel())
    return adj_syn


def index2mask(index, size):
    """
    Convert an index list to a boolean mask.

    Parameters
    ----------
    index : list or tensor
        List or tensor of indices to be set to True.
    size : int or tuple of int
        Shape of the mask. If an integer, the mask is 1-dimensional.

    Returns
    -------
    mask : tensor
        A boolean tensor of the specified size, with True at the given `index` positions and False elsewhere.

    Examples
    --------
    >>> index = [0, 2, 4]
    >>> size = 5
    >>> index2mask(index, size)
    tensor([True, False, True, False, True], dtype=torch.bool)
    """
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def splits(data, exp='default'):
    # customize your split here
    if hasattr(data, 'y'):
        num_classes = max(data.y) + 1
    else:
        num_classes = max(data.labels_full).item() + 1
    # data.nclass = num_classes
    if not hasattr(data, 'train_mask'):
        indices = []
        for i in range(num_classes):
            data.y = data.y.reshape(-1)
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        elif exp == 'few':
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)
        else:
            # if fixed but no split is provided, use the default 8/1/1 split classwise
            train_index = torch.cat([i[:int(i.shape[0] * 0.8)] for i in indices], dim=0)
            val_index = torch.cat([i[int(i.shape[0] * 0.8):int(i.shape[0] * 0.9)] for i in indices], dim=0)
            test_index = torch.cat([i[int(i.shape[0] * 0.9):] for i in indices], dim=0)
            # raise NotImplementedError('Unknown split type')
        data.train_mask = index2mask(train_index, size=data.num_nodes)
        data.val_mask = index2mask(val_index, size=data.num_nodes)
        data.test_mask = index2mask(test_index, size=data.num_nodes)
    data.idx_train = data.train_mask.nonzero().view(-1)
    data.idx_val = data.val_mask.nonzero().view(-1)
    data.idx_test = data.test_mask.nonzero().view(-1)

    return data


def save_reduced(adj_syn=None, feat_syn=None, labels_syn=None, args=None):
    if not os.path.exists(args.syn_path):
        os.makedirs(args.syn_path)
    if adj_syn is not None:
        torch.save(adj_syn,
                   f'{args.syn_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
    if feat_syn is not None:
        torch.save(feat_syn,
                   f'{args.syn_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
    if labels_syn is not None:
        torch.save(labels_syn,
                   f'{args.syn_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
    args.logger.info(f"Saved {args.syn_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt")


def load_reduced(args, data=None):
    flag = 0
    try:
        feat_syn = torch.load(
            f'{args.syn_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
    except:
        print("find no feat, use original feature matrix instead")
        flag += 1
        if args.setting == 'trans':
            feat_syn = data.feat_full
        else:
            feat_syn = data.feat_train
    try:
        labels_syn = torch.load(
            f'{args.syn_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
    except:
        print("find no label, use original label matrix instead")
        flag += 1
        labels_syn = data.labels_train

    try:
        adj_syn = torch.load(
            f'{args.syn_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
    except:
        print("find no adj, use identity matrix instead")
        flag += 1
        adj_syn = torch.eye(feat_syn.size(0), device=args.device)
    args.logger.info(f"Load {args.syn_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt")
    assert flag < 3, "no file found, please run the reduction method first"

    return adj_syn, feat_syn, labels_syn

def get_syn_data(data, args, model_type, verbose=False):
    """
    Loads or computes synthetic data for evaluation.

    Parameters
    ----------
    data : Dataset
        The dataset containing the graph data.
    model_type : str
        The type of model used for generating synthetic data.
    verbose : bool, optional, default=False
        Whether to print detailed logs.

    Returns
    -------
    feat_syn : torch.Tensor
        Synthetic feature matrix.
    adj_syn : torch.Tensor
        Synthetic adjacency matrix.
    labels_syn : torch.Tensor
        Synthetic labels.
    """
    adj_syn, feat_syn, labels_syn = load_reduced(args, data)

    if labels_syn.shape[0] == data.labels_train.shape[0]:
        return feat_syn, adj_syn, labels_syn

    if type(adj_syn) == torch.tensor and is_sparse_tensor(adj_syn):
        adj_syn = adj_syn.to_dense()
    elif isinstance(adj_syn, torch.sparse.FloatTensor):
        adj_syn = adj_syn.to_dense()
    else:
        adj_syn = adj_syn

    adj_syn = sparsify(model_type, adj_syn, args, verbose=verbose)
    return feat_syn, adj_syn, labels_syn