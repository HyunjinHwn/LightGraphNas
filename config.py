'''Configuration'''
import os
import sys
import pdb
import json
import logging
import torch

import click
from pprint import pformat
from project.utils import seed_everything, f1_macro, accuracy, roc_auc

representative_r = {
    'cora': 0.5,
    'citeseer': 0.5,
    'pubmed': 0.5,
    'flickr': 0.01,
    'reddit': 0.001,
    'ogbn-arxiv': 0.01,
    'yelp': 0.001,
    'amazon': 0.002,
    'pubmed': 0.1
}

class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __repr__(self):
        # Use pprint's pformat to print the dictionary in a pretty manner
        return pformat(self.__dict__, compact=True)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)


def update_from_dict(obj, updates):
    for key, value in updates.items():
        # set higher priority from command line as we explore some factors
        if key in ['init'] and obj.init is not None:
            continue
        setattr(obj, key, value)

def systematic_config(args):
    args.cwd = os.getcwd()
    
    if args.gpu_id >= 0:
        args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = 'cpu'
        
    if args.reduction_rate == -1:
        args.reduction_rate = representative_r[args.dataset]
    
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'pubmed']:
        args.setting = 'trans'
    elif args.dataset in ['flickr', 'reddit', 'amazon','yelp']:
        args.setting = 'ind'
        
    args.metric = f1_macro if args.dataset in ['yelp', 'amazon'] else accuracy
    
    if args.eval_interval == -1: 
        args.eval_interval = args.epochs // 10
    args.checkpoints = list(range(-1, args.epochs + 1, args.eval_interval))
    
    if args.eval_model == None:
        args.eval_model = args.condense_model
    if args.final_eval_model == None:
        args.final_eval_model = args.condense_model
    if args.condense_model == 'APPNP':
        args.K = 10
    elif args.condense_model == 'Cheby':
        args.K = 2
    elif args.condense_model == 'GraphSageSample':
        args.sage_samples = [25] * args.nlayers
    return args

@click.command()
# ===============================overall execution settings================================ #
@click.option('--mode',  default=None, help='mode for excution: grid_search, condense, nas')
@click.option('--gpu_id', '-G', default=0, help='gpu id start from 0, -1 means cpu', show_default=True)
@click.option('--seed', '-S', default=1, help='Random seed', show_default=True)
@click.option('--verbose', '-V', is_flag=True, show_default=True)
@click.option('--root', '--sp', default='/data/condensation', show_default=True, help='save path for synthetic graph')
@click.option('--data_path', '--lp', default='/data/condensation/data', show_default=True, help='data loading path')

# ================================dataset settings================================ #
@click.option('--dataset', '-D', default='cora', show_default=True)
@click.option('--setting', type=click.Choice(['trans', 'ind']), show_default=True,
              help='transductive or inductive setting')
@click.option('--split', default='fixed', show_default=True,
              help='only support public split now, do not change it')  # 'fixed', 'random', 'few'
@click.option('--pre_norm', default=True, show_default=True,
              help='pre-normalize features, forced true for arxiv, flickr and reddit')
@click.option('--multi_label', is_flag=True, show_default=True) # Never call True

# ================================condense model settings================================ #
@click.option('--condense_model', default='None',
              type=click.Choice(
                  ['None', 'Linear', 'GCN', 'GAT', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 
                   'GraphSageSample', 'Parallel','ParallelWeightV1','ParallelWeightV2', 'Original']
              ), show_default=True)
# @click.option('--multi_model',  default=None)
@click.option('--hidden', '-H', default=256, show_default=True)
@click.option('--nlayers', default=2, help='number of GNN layers of condensed model', show_default=True)
@click.option('--activation', default='relu', help='activation function, only for APPNP', show_default=True,
              type=click.Choice(['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']))
# model specific args
@click.option('--alpha', default=0.1, help='for appnp', show_default=True)
@click.option('--K', default=10, help='for appnp', show_default=True)

# ===============================condense method settings================================ #
@click.option('--method', '-M', default='kcenter', show_default=True,
              type=click.Choice(
                   ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC',
                   'affinity_GS', 'kron', 'vng', 'clustering', 'averaging',
                   'gcond', 'doscond', 'gcondx', 'doscondx', 'gcdm', 'sfgc', 'msgc', 'disco', 'sgdd', 'gcsntk', 'geom','cadm',
                   'cent_d', 'cent_p', 'kcenter', 'herding', 'random', 'random_edge',
                   'original']))
@click.option('--reduction_rate', '-R', default=-1.0, show_default=True,
              help='-1 means use representative reduction rate; reduction rate of training set, defined as (number of nodes in small graph)/(number of nodes in original graph)')
@click.option('--init', default='random', help='features initialization methods', show_default=True,
              type=click.Choice(['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 
                                'algebraic_JC', 'affinity_GS', 'kron', 'vng', 'clustering', 'averaging',
                                'cent_d', 'cent_p', 'kcenter', 'herding', 'random']))
@click.option('--dis_metric', default='mse', show_default=True,
              help='distance metric for all condensation methods, ours means metric used in GCond paper')

# ================================condense optimization settings================================ #
@click.option('--epochs', '-E', default=1000, show_default=True, help='number of reduction epochs')
@click.option('--outer_loop', default=10, show_default=True)
@click.option('--inner_loop', default=1, show_default=True)
@click.option('--lr_adj', default=1e-4, show_default=True)
@click.option('--lr_feat', default=1e-4, show_default=True)
@click.option('--optim', default="Adam", show_default=True)
@click.option('--threshold', default=0.0, show_default=True, help='sparsificaiton threshold before evaluation')
@click.option('--cond_dropout', default=0.0, show_default=True)
@click.option('--ntrans', default=1, show_default=True, help='number of transformations in SGC and APPNP')

# ================================condense evaluation settings================================ #
@click.option('--run_final_eval', default=3, show_default=True, help='repeat times of final evaluations')
@click.option('--run_inter_eval', default=3, show_default=True, help='repeat times of intermediate evaluations')
@click.option('--eval_interval', default=-1, show_default=True, help='eval interval during condensation, -1 means 1/10 of epochs')
@click.option('--eval_epochs', '--ee', default=300, show_default=True)
@click.option('--eval_loss', '--eloss', default='CE',type=click.Choice(['CE', 'KLD','MSE']), show_default=True)
@click.option('--lr', default=0.01, show_default=True)
@click.option('--eval_dropout', default=0.3, show_default=True)
@click.option('--weight_decay', '--wd', default=0.0, show_default=True)
@click.option('--eval_model', default=None, show_default=True)
@click.option('--final_eval_model', default=None, show_default=True)

# ================================other settings================================ #
# coarsening settings
@click.option('--coarsen_strategy', '--cs', default='greedy', help='for edge contraction method',
              type=click.Choice(['optimal', 'greedy']), show_default=True)
# coreset settings
@click.option('--agg', is_flag=True, show_default=True, help='use aggregation for coreset methods')
# # sfgc
# @click.option('--lr_teacher',  default=1.0)
# @click.option('--lr_student',  default=1.0)
# @click.option('--wd_teacher',  default=1.0)
# @click.option('--teacher_epochs',  default=1)
# @click.option('--expert_epochs',  default=1)
# @click.option('--syn_steps',  default=1)
# @click.option('--start_epoch',  default=1)
# @click.option('--no_buff', is_flag=True, show_default=True, help='skip the buffer generation and use existing in geom,sfgc')
# # simgc
# @click.option('--feat_alpha', default=10, show_default=True, help='feature loss weight')
# @click.option('--smoothness_alpha', default=0.1, show_default=True, help='smoothness loss weight')
# # gdem
# @click.option('--eigen_k', default=60, show_default=True, help='number of eigenvalues')
# @click.option('--ratio', default=0.8, show_default=True, help='eigenvalue loss weight')
# @click.option('--lr_eigenvec', default=0.01, show_default=True, help='eigenvalue loss weight')
# @click.option('--gamma', default=0.5, show_default=True, help='eigenvalue loss weight')
# # geom
# @click.option('--soft_label', default=0, show_default=True)
# # gcsntk
# @click.option('--mx_size', default=100, help='for gcsntk methods, avoid SVD error', show_default=True)
# # msgc
# @click.option('--batch_adj', default=16, show_default=True, help='batch size for msgc')
# # t_spanner
# @click.option('--ts', default=4, help='for tspanner', show_default=True)
# later
# @click.option('--with_structure', default=1, show_default=True, help='if synthesizing structure in PropertyEvaluator')


@click.pass_context
def cli(ctx, **kwargs):
    args = dict2obj(kwargs)
    args = systematic_config(args)
    
    ## condensation setting
    if args.mode == 'grid_search':    
        # use received args
        config_path = None
        pass
    elif args.mode == 'condense':
        # use args from config file
        if args.method != 'Original' and args.method not in ['kcenter', 'herding', 'random', 'cent_p', 'cent_d']:
            config_path = f"{os.path.join(args.cwd, 'configs', args.dataset, args.method, args.condense_model)}.json"
            try:
                conf_dt = json.load(open(config_path))
                print("Loaded config file:", config_path)
                update_from_dict(args, conf_dt)
            except:
                print("No config file found in {}".format(config_path))
                exit()
       
    elif args.mode == 'nasmy':
        args.mode = 'condense'
        pass
    
    elif args.mode == 'nasappnp':
        args.mode = 'condense'
        pass
    
    elif args.mode == 'original':
        pass
    
    elif 'eval' in args.mode:
        pass
    else:
        raise NotImplementedError(f"Not implemented mode: {args.mode}")
    
    args.syn_path = f"{os.path.join(args.root, args.mode, args.dataset, args.method, args.condense_model)}"
    args.log_path = f"{os.path.join(args.cwd, 'logs', args.mode, args.dataset, args.method, args.condense_model)}.log"
    if args.mode == 'original':
        args.log_path = f"{os.path.join(args.cwd, 'logs', args.mode, args.dataset)}.log"
    if 'eval' in args.mode:
        args.syn_path = f"{os.path.join(args.root, 'condense', args.dataset, args.method, args.condense_model)}"
        
    os.makedirs(os.path.dirname(os.path.dirname(os.path.dirname(args.log_path))), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.dirname(args.log_path)), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("[Terminal] %(levelname)s: %(message)s"))
    console_handler.setLevel(logging.INFO)  # INFO 이상 출력
    file_handler = logging.FileHandler(args.log_path, mode='a')
    file_handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s"))
    file_handler.setLevel(logging.WARNING)  # WARNING 이상만 파일에 기록

    logger = logging.getLogger("custom_logger")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # logger.debug("DEBUG: 이건 기본적으로 표시 안 됨")   # DEBUG < INFO → 출력 안 됨
    # logger.info("INFO: 터미널만 출력됩니다.")           # 콘솔만
    # logger.warning("WARNING: 터미널 + 로그파일 기록됩니다.")  # 둘 다
    # logger.error("ERROR: 터미널 + 로그파일 기록됩니다.")      # 둘 다
    args.logger = logger
    args.logger.warning(f"{args.mode}, {args.dataset}, {args.method}, {args.condense_model}")
    args.logger.warning(f"Config: lr_feat {args.lr_feat} lr_adj {args.lr_adj} dis_metric {args.dis_metric} " 
                        + f"outer_loop {args.outer_loop} inner_loop {args.inner_loop}")
    args.logger.warning("Intermediate evaluation on "+ str(args.checkpoints))
    return args


def get_args():
    return cli(standalone_mode=False)


if __name__ == '__main__':
    cli()
