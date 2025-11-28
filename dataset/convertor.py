# from deeprobust.graph.data import Dataset
from typing import Optional

import numpy as np
import torch
from pygsp import graphs
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_undirected, to_dense_adj, remove_self_loops, add_self_loops
from torch_sparse import SparseTensor
import networkit as nk
from torch_geometric.data import Data, HeteroData
# import dgl


def from_dgl(g, name, hetero=True):
    if g.is_homogeneous:
        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value

        return data

    data = HeteroData()
    data.name = name

    for node_type in g.ntypes:
        for attr, value in g.nodes[node_type].data.items():
            data[node_type][attr] = value

    for edge_type in g.canonical_etypes:
        row, col = g.edges(form="uv", etype=edge_type)
        data[edge_type].edge_index = torch.stack([row, col], dim=0)
        for attr, value in g.edge_attr_schemes(edge_type).items():
            data[edge_type][attr] = value

    data_out = Data()
    if not hetero:
        edge_index_list = []
        for edge_type in g.canonical_etypes:
            edge_index_list.append(data[edge_type].edge_index)
        data_out.edge_index = add_self_loops(torch.cat(edge_index_list, dim=1))[0]
        data_out.x = data.node_stores[0]['feature']  # Features for each node
        # Assigning labels to data_out
        data_out.y = data.node_stores[0]['label']  # Labels for each node

        # Assuming the train, validation, and test masks are also in node_stores[0]
        #data_out.train_mask = data.node_stores[0]['train_mask']  # Training mask
        #data_out.val_mask = data.node_stores[0].get('val_mask', None)  # Optional: Validation mask (if exists)
        #data_out.test_mask = data.node_stores[0]['test_mask']  # Test mask

    data_out.num_nodes = len(data_out.x)
    data_out.num_classes = max(data_out.y).item() + 1

    return data_out
def pyg2gsp(edge_index):
    G = graphs.Graph(W=to_dense_adj(to_undirected(edge_index))[0])
    return G


def csr2ei(adjacency_matrix_csr):
    adjacency_matrix_coo = adjacency_matrix_csr.tocoo()
    # Convert numpy arrays directly to a tensor
    edge_index = torch.tensor(np.vstack([adjacency_matrix_coo.row, adjacency_matrix_coo.col]), dtype=torch.long)
    return edge_index


def ei2csr(edge_index, num_nodes):
    edge_index = edge_index.numpy()
    scoo = coo_matrix((np.ones_like(edge_index[0]), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    adjacency_matrix_csr = scoo.tocsr()
    return adjacency_matrix_csr


def dense2sparsetensor(mat: torch.Tensor, has_value: bool = True):
    if mat.dim() > 2:
        index = mat.abs().sum([i for i in range(2, mat.dim())]).nonzero()
    else:
        index = mat.nonzero()
    index = index.t()

    row = index[0]
    col = index[1]

    value: Optional[torch.Tensor] = None
    if has_value:
        value = mat[row, col]

    return SparseTensor(
        row=row,
        rowptr=None,
        col=col,
        value=value,
        sparse_sizes=(mat.size(0), mat.size(1)),
        is_sorted=True,
        trust_data=True,
    )


def networkit_to_pyg(graph):
    # Extract edges from Networkit graph
    edges = list(graph.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Check if the graph is weighted
    if graph.isWeighted():
        edge_attr = torch.tensor([graph.weight(u, v) for u, v in edges], dtype=torch.float)
    else:
        edge_attr = None

    pyg_graph = Data(edge_index=edge_index, edge_attr=edge_attr)
    return pyg_graph


def pyg_to_networkit(pyg_graph):
    # Create an empty Networkit graph
    # if hasattr(pyg_graph, 'edge_attr') and pyg_graph.edge_attr is not None:
    #     graph = nk.Graph(weighted=True, directed=False)
    # else:
    #     graph = nk.Graph(weighted=False, directed=False)

    # Add edges to the Networkit graph
    edge_index = pyg_graph.edge_index.numpy()
    if hasattr(pyg_graph, 'edge_attr') and pyg_graph.edge_attr is not None:
        edge_attr = pyg_graph.edge_attr.numpy()
        graph = nk.GraphFromCoo(inputData=(edge_attr, (edge_index[0], edge_index[1])), n=pyg_graph.num_nodes,
                                weighted=True, directed=False)
    else:
        graph = nk.GraphFromCoo(inputData=((edge_index[0], edge_index[1])), n=pyg_graph.num_nodes,
                                weighted=False, directed=False)

    graph.indexEdges()

    return graph
