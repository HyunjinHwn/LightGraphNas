import csv
import os
import pickle as pkl
from itertools import product
from pathlib import Path

from scipy.stats import pearsonr
from tqdm import tqdm
import numpy as np
import torch

# from project.evaluation.eval_agent import Evaluator
from project.dataset import *
from .nas_model_my import TwoLayerNet
from project.utils import seed_everything


def save_csv(file_path, num):
    file_path = Path(file_path)
    with file_path.open(mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(num)
    print("saved csv", file_path)


def load_csv(file_path):
    file_path = Path(file_path)
    with file_path.open(mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


def save_pkl(file_path, data):
    file_path = Path(file_path)
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)


def load_pkl(file_path):
    file_path = Path(file_path)
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


class NasEvaluatorMy:
    """
    Class for evaluating neural architecture search (NAS) performance on original and synthetic graphs.
    """
    def __init__(self, args, save_path, save_path_ori):
        self.args = args
        self.save_path = save_path
        self.save_path_ori = save_path_ori
        self.run_evaluation = 1
        self.best_params_syn, self.best_params_ori = None, None
        self.results_syn, self.results_ori = [], []
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        # Define possible values for parameters to search over
        # 7*7*8 = 392 combinations
        layers1 = ['SGC', 'GCN', 'GAT', 'GraphSageSample', 'Cheby', 'APPNP', 'Linear']
        layers2 = ['SGC', 'GCN', 'GAT', 'GraphSageSample', 'Cheby', 'APPNP', 'Linear']
        activations = ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']

        # ks = [2, 4, 6]
        # nhids = [16, 32]
        # alphas = [0.1]
        # activations = ['relu']

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.parameter_combinations = list(product(layers1, layers2, activations))

    def evaluate(self, data, model_type, verbose=False, reduced=None):
        layer1, layer2, act = model_type
        args = self.args
        res = []
        # Prepare synthetic data if required
        data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type="SGC", verbose=verbose)
        run_evaluation = range(self.run_evaluation)
        
        # Collect validation accuracy results from multiple runs
        val_results,test_results  = [], []
        for i in run_evaluation:
            seed_everything(args.seed + i)
            model = TwoLayerNet(data.feat_syn.shape[1], args.hidden, data.nclass, args, mode='eval',
                                layer1=layer1, layer2=layer2, activation=act).to(self.device)
            best_acc_val = model.fit_with_val(data, train_iters=args.eval_epochs, verbose=verbose,
                                              setting=args.setting, reduced=reduced)
            model.eval()
            labels_test = data.labels_test.long().to(args.device)
        
            output = model.predict(data.feat_full, data.adj_full)
            # loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = args.metric(output[data.idx_test], labels_test).item()
            
            test_results.append(acc_test)
            
        test_results = np.array(test_results)
        return test_results.mean(), test_results.std()
    
    
    def nas_evaluate(self, data, model_type, verbose=False, reduced=None):
        layer1, layer2, act = model_type
        args = self.args
        res = []
        # Prepare synthetic data if required
        data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type="SGC", verbose=verbose)
        run_evaluation = range(self.run_evaluation)

        # Collect validation accuracy results from multiple runs
        for i in run_evaluation:
            model = TwoLayerNet(data.feat_syn.shape[1], args.hidden, data.nclass, args, mode='train',
                                layer1=layer1, layer2=layer2, activation=act).to(self.device)
            
            best_acc_val = model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                                              setting=args.setting, reduced=reduced)
            res.append(best_acc_val)
        res = np.array(res)

        return res.mean(), res.std()
    
    def evaluate_ori(self, data):
        """
        Evaluates various architectures on the original graph and identifies the best one.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        """
        best_acc_val_ori = 0
        file_path_csv = f'{self.save_path}/results_ori.csv'
        file_path = f'{self.save_path}/best_params_ori.pkl'
        for params in tqdm(self.parameter_combinations):
            acc_val_ori, _ = self.nas_evaluate(data, model_type=params, reduced=False, verbose=False)
            self.results_ori.append(list(params)+[str(acc_val_ori)])

            # Update best architecture based on validation accuracy
            if acc_val_ori > best_acc_val_ori:
                best_acc_val_ori = acc_val_ori
                self.best_params_ori = params

            # Save results to files
            save_csv(file_path_csv, self.results_ori[-1])
            save_pkl(file_path, self.best_params_ori)
        

    def evaluate_syn(self, data):
        """
        Evaluates various architectures on the synthetic graph and identifies the best one.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        """
        best_acc_val_syn = 0
        file_path_csv = f'{self.save_path}/{self.args.method}_{self.args.condense_model}_results_syn.csv'
        file_path = f'{self.save_path}/{self.args.method}_{self.args.condense_model}_best_params_syn.pkl'
        for params in tqdm(self.parameter_combinations):
            acc_val_syn, _ = self.nas_evaluate(data, model_type=params, reduced=True, verbose=False)
            self.results_syn.append(list(params)+[str(acc_val_syn)])

            # Update best architecture based on validation accuracy
            if acc_val_syn > best_acc_val_syn:
                best_acc_val_syn = acc_val_syn
                self.best_params_syn = params

            # Save results to files
            save_csv(file_path_csv, self.results_syn[-1])
            save_pkl(file_path, self.best_params_syn)

    def test_params_ori(self, data):
        """
        Tests the best architecture on the original graph using the best parameters.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        """
        if self.best_params_ori is None:
            file_path = f'{self.save_path}/best_params_ori.pkl'
            self.best_params_ori = load_pkl(file_path)

        self.args.logger.info(f"Best parameters for original graph: {self.best_params_ori}")

        acc_test_ori, _ = self.evaluate(data, model_type=self.best_params_ori, reduced=False, verbose=False)
        self.args.logger.info(f"Test accuracy on original graph: {acc_test_ori}")
        return self.best_params_ori, acc_test_ori

    def test_params_syn(self, data):
        """
        Tests the best architecture on the synthetic graph using the best parameters.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.

        Returns
        -------
        acc_test_syn : float
            The test accuracy on the synthetic graph.
        """
        if self.best_params_syn is None:
            file_path = f'{self.save_path}/{self.args.method}_{self.args.condense_model}_best_params_syn.pkl'
            self.best_params_syn = load_pkl(file_path)

        self.args.logger.info(f"Best parameters for synthetic graph: {self.best_params_syn}")

        acc_test_syn, _ = self.evaluate(data, model_type=self.best_params_syn, reduced=True, verbose=False)
        self.args.logger.info(f"Test accuracy on synthetic graph: {acc_test_syn}")

        return acc_test_syn

    def get_rank(self, results):
        """
        Ranks results based on their values.

        Parameters
        ----------
        results : list of float
            The list of results to rank.

        Returns
        -------
        ranks : list of int
            The list of ranks corresponding to the results.
        """
        sorted_tuples = sorted(enumerate(results), key=lambda x: x[1], reverse=True)
        rank_count = 1

        rank_dict = {}
        for _, value in sorted_tuples:
            if value not in rank_dict:
                rank_dict[value] = rank_count
            rank_count += 1

        ranks = [rank_dict[value] for value in results]

        return ranks

    def cal_pearson(self):
        """
        Calculates Pearson correlation coefficients between synthetic and original results.

        Returns
        -------
        pearson_corr_acc : float
            Pearson correlation coefficient of accuracies.
        pearson_corr_rank : float
            Pearson correlation coefficient of ranks.
        """
        results_syn = load_csv(f'{self.save_path}/{self.args.method}_{self.args.condense_model}_results_syn.csv')
        self.results_syn = [float(x[-1]) for x in results_syn]
        self.results_ori = [float(x[-1]) for x in load_csv(f'{self.save_path_ori}/results_ori.csv')]

        pearson_corr_acc, _ = pearsonr(self.results_syn, self.results_ori)
        self.args.logger.info(f"Pearson correlation of accuracy: {pearson_corr_acc:.3f}")

        results_syn_ranked = self.get_rank(self.results_syn)
        results_ori_ranked = self.get_rank(self.results_ori)
        pearson_corr_rank, _ = pearsonr(results_syn_ranked, results_ori_ranked)
        self.args.logger.info(f"Pearson correlation of rank: {pearson_corr_rank:.3f}")

        return pearson_corr_acc, pearson_corr_rank
