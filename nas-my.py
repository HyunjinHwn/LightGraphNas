import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from project.config import cli
from project.dataset import *
from project.evaluation import NasEvaluatorMy
## python nas-my.py --dataset cora --method gcond --condense_model Linear --gpu_id 0 --mode nasmy &
## python nas-my.py --dataset cora --method gcond --condense_model SGC --gpu_id 1 --mode nasmy &
## python nas-my.py --dataset cora --method gcond --condense_model GCN --gpu_id 2 --mode nasmy &
## python nas-my.py --dataset cora --method gcond --condense_model GAT --gpu_id 3 --mode nasmy &
## python nas-my.py --dataset cora --method gcond --condense_model GraphSage --gpu_id 4 --mode nasmy &
## python nas-my.py --dataset cora --method gcond --condense_model APPNP --gpu_id 5 --mode nasmy &
## python nas-my.py --dataset cora --method gcond --condense_model Cheby --gpu_id 6 --mode nasmy &
## python nas-my.py --dataset cora --method gcond --condense_model Parallel --gpu_id 6 --mode nasmy

if __name__ == '__main__':
    args = cli(standalone_mode=False)


    data = get_dataset(args.dataset, args)


    save_path_ori = f'logs/nasmy/{args.dataset}'
    os.makedirs(save_path_ori, exist_ok=True)
    if not os.path.exists(f'{save_path_ori}/results_ori.csv'):
        NasEval = NasEvaluatorMy(args, save_path_ori, save_path_ori)
        args.logger.info("No original results found. Run evaluate_ori and test_params_ori.")
        NasEval.evaluate_ori(data)
        best_params_ori, acc_test_ori = NasEval.test_params_ori(data)
    else:
        args.logger.info("Find original results. Run evaluate_syn and test_params_syn.")
        NasEval = NasEvaluatorMy(args, save_path_ori, save_path_ori)
        best_params_ori, acc_test_ori = NasEval.test_params_ori(data)
    if not os.path.exists(f'{save_path_ori}/results_ori.log'):
        with open(f'{save_path_ori}/results_ori.log', 'w') as f:
            f.write(f"Best Params: {best_params_ori}\n")
            f.write(f"Test Acc: {acc_test_ori}\n")

    save_path = f'logs/nasmy/{args.dataset}/{args.method}/{args.condense_model}'
    NasEval = NasEvaluatorMy(args, save_path, save_path_ori=save_path_ori)
    if not os.path.exists(f'{save_path}/{args.method}_{args.condense_model}_results_syn.csv'):
        args.logger.info("No synthetic results found. Run evaluate_syn and test_params_syn.")
        NasEval.evaluate_syn(data)
    print("Config", args.dataset, args.method, args.condense_model)
    with open(f'{save_path}/results.log', 'w') as f:
        f.write(f"Best Params Syn: {NasEval.best_params_syn}\n")
        res_acc = NasEval.test_params_syn(data)
        f.write(f"Test Acc Syn: {res_acc}\n")
        res_person_acc, res_person_rank = NasEval.cal_pearson()
        f.write(f"Pearson Acc Syn: {res_person_acc}\n")
        f.write(f"Pearson Rank Syn: {res_person_rank}\n")
