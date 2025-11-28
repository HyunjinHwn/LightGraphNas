import numpy as np
import torch
import torch.nn.functional as F

from LightGraphNas.dataset import *
from LightGraphNas.evaluation import *
from LightGraphNas.evaluation.utils import *
from LightGraphNas.models import *
from LightGraphNas.utils import accuracy, seed_everything


class Evaluator:
    """
    A class to evaluate different models and their hyperparameters on graph data.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments and configuration parameters.
    **kwargs : keyword arguments
        Additional parameters.
    """

    def __init__(self, args, **kwargs):
        """
        Initializes the Evaluator with given arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Command-line arguments and configuration parameters.
        **kwargs : keyword arguments
            Additional parameters.
        """
        self.args = args
        self.device = args.device
        self.metric = args.metric
        
    
    def evalOnGraph(self, data, model_type, verbose=True, reduced=True, train='eval'):
        """
        Evaluates a model over multiple runs and returns mean and standard deviation of accuracy.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model to evaluate.
        verbose : bool, optional, default=True
            Whether to print detailed logs.
        reduced : bool, optional, default=True
            Whether to use synthetic data.
        mode : str, optional, default='eval'
            The mode for the model (e.g., 'eval' or 'cross').

        Returns
        -------
        mean_acc : float
            Mean accuracy over multiple runs.
        std_acc : float
            Standard deviation of accuracy over multiple runs.
        """

        args = self.args

        # Prepare synthetic data if required
        if reduced:
            data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type=model_type,
                                                                        verbose=verbose)

        # Collect accuracy results from multiple runs
        val_results, test_results  = [], []
        for i in range(args.run_final_eval):
            seed_everything(args.seed + i)
            model = eval(model_type)(data.feat_full.shape[1], args.hidden, data.nclass, args).to(self.device)
            
            best_acc_val = model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, 
                                              verbose=verbose, setting=args.setting, reduced=reduced)
            best_test_acc = self.test(data, model, verbose=args.verbose)
            
            if type(best_acc_val) == torch.Tensor:
                best_acc_val = best_acc_val.cpu().item()
            val_results.append(best_acc_val)        
            test_results.append(best_test_acc)
        val_results = np.array(val_results)
        test_results = np.array(test_results)
        
        # Log and return mean and standard deviation of accuracy
        args.logger.warning(f'Model: {model_type}, Seed:{args.seed}, '+
                         f'Vaild Mean Accuracy: {100 * val_results.mean():.2f} +/- {100 * val_results.std():.2f}, '+
                         f'Test Mean Accuracy: {100 * test_results.mean():.2f} +/- {100 * test_results.std():.2f}')
        return test_results.mean(), test_results.std()


    def test(self, data, model, verbose=True):
        """
        Tests a model and returns accuracy and loss.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.

        Returns
        -------
        acc_test : float
            Accuracy on test set.
        """
        args = self.args

        model.eval()
        labels_test = data.labels_test.long().to(args.device)

        if args.setting == 'ind':
            output = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = self.metric(output, labels_test).item()

            if verbose:
                print("Test set results:",
                      f"loss= {loss_test.item():.4f}",
                      f"accuracy= {acc_test:.4f}")
        else:
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = self.metric(output[data.idx_test], labels_test).item()
            if verbose:
                print("Test full set results:", f"loss= {loss_test.item():.4f}", f"accuracy= {acc_test:.4f}")
        return acc_test
