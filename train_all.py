import pdb
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
import torch
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
from project.config import cli
from project.dataset import *
from project.evaluation import *
from project.utils import seed_everything, to_tensor
from project.sparsification import *
from project.condensation import CallAgent

# python train_all.py -M doscond -D cora --condense_model Parallel --mode condense --gpu_id 7 &
# python train_all.py -M doscond -D cora --condense_model ParallelWeightV1 --mode grid_search --gpu_id 0
if __name__ == '__main__':
    args = cli(standalone_mode=False)
    graph = get_dataset(args.dataset, args)
    seed_everything(args.seed)
    
    
    agent = CallAgent(graph, args)
    reduced_graph = agent.reduce(graph, verbose=args.verbose)
    if reduced_graph.feat_syn.device != args.device:
        if reduced_graph.feat_syn!= None: reduced_graph.feat_syn = reduced_graph.feat_syn.to(args.device)
        if reduced_graph.adj_syn != None: 
            if isinstance(reduced_graph.adj_syn, torch.Tensor):
                reduced_graph.adj_syn = reduced_graph.adj_syn.to(args.device)
            else:
                # sparse tensor
                reduced_graph.adj_syn = to_tensor(reduced_graph.adj_syn, device=args.device)
        reduced_graph.labels_syn = reduced_graph.labels_syn.to(args.device)
    
    
    evaluator = Evaluator(args)
    for eval_model in [ 'SGC', 'GCN', 'GAT', 'GraphSage', 'GraphSageSample', 'Cheby', 'APPNP', 'Linear']: # SGC GCN GAT APPNP Cheby GraphSage
        all_res = evaluator.evalOnGraph(graph, model_type=eval_model, verbose=False, reduced=True)
