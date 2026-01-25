from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import tempfile

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr, rankdata
import matplotlib.pyplot as plt

from cycling_predictor.models import BaseModel
from cycling_predictor.predictions import CPPrediction


class XGBModel(BaseModel):
    """
    CyclingPredictor XGBModel class.
    """
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    @property
    def model_type(self) -> str:
        return 'XGB'

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            'k': 20,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'learning_rate': 0.05,
            'max_depth': 5,
            'reg_alpha': 3,
            'reg_lambda': 1,
            'n_estimators': 1000
        }

    @property
    def dump_fn(self) -> str:
        return (
            f"{self.name}_{self.config.get('k')}_{self.config.get('learning_rate')}_{self.config.get('max_depth')}_"
            f"{self.config.get('reg_alpha')}_{self.config.get('reg_lambda')}_{self.config.get('n_estimators')}.json")

    def train(self, x: np.ndarray, group_sizes: List[int], y: np.ndarray, verbose: Optional[bool] = True) -> None:
        """
        Train the XGB model.

        :param x: Input training features.
        :param group_sizes: Sizes of groups in the training set.
        :param y: Output training ranks.
        :param verbose: Whether to print the training results.
        """

        # Set up the model
        xgb_ranker = xgb.XGBRanker(
            objective='rank:ndcg',
            eval_metric=f'ndcg@{self.config.get('k')}',
            **self.config)

        # Convert y_train for NDCG ranking
        yr_train = self._convert_ranks_to_relevance(y, group_sizes, _k=self.config.get('k'))

        # Train the model
        xgb_ranker.fit(x, yr_train, group=group_sizes)

        # Assign the model
        self.model = xgb_ranker

    def test(self, x: np.ndarray, group_sizes: List[int], y: np.ndarray, verbose: Optional[bool] = True) -> None:
        """
        Test the XGB model.

        :param x: Input testing features.
        :param group_sizes: Sizes of groups in the testing set.
        :param y: Output testing ranks.
        :param verbose: Whether to print the test results.
        """

        # Make predictions
        predictions = self.model.predict(x)

        # Process predictions per group
        start = 0
        sr10ps, sr20ps, sr10rs, sr20rs = list(), list(), list(), list()
        for group_size in group_sizes:

            # Define end and retrieve sorted prediction
            end = start + group_size

            # Create prediction
            prediction_obj = CPPrediction(
                prediction=rankdata(-predictions[start:end], method='ordinal'),
                scores=predictions[start:end],
                result=y[start:end] if y is not None else None,
            )

            # Compute metrics and append
            sr10p = prediction_obj.spearmanr_k_prediction(10)
            sr20p = prediction_obj.spearmanr_k_prediction(20)
            sr10ps.append(sr10p)
            sr20ps.append(sr20p)
            sr10r = prediction_obj.spearmanr_k_result(10)
            sr20r = prediction_obj.spearmanr_k_result(20)
            sr10rs.append(sr10r)
            sr20rs.append(sr20r)

            # Shift start
            start = end

        # Compute metrics
        sr10p_avg = sum(sr10ps) / len(sr10ps)
        sr10p_std = np.std(sr10ps)
        sr20p_avg = sum(sr20ps) / len(sr20ps)
        sr20p_std = np.std(sr20ps)
        sr10r_avg = sum(sr10rs) / len(sr10rs)
        sr10r_std = np.std(sr10rs)
        sr20r_avg = sum(sr20rs) / len(sr20rs)
        sr20r_std = np.std(sr20rs)
        sr10_avg = (sr10p_avg + sr10r_avg) / 2
        sr20_avg = (sr20p_avg + sr20r_avg) / 2

        # Print metrics if verbose
        if verbose:
            print(f"\nAvg. Spearman's Rho pred. (top 10): {sr10p_avg} ± {sr10p_std}")
            print(f"Avg. Spearman's Rho pred. (top 20): {sr20p_avg} ± {sr20p_std}")
            print(f"Avg. Spearman's Rho res. (top 10): {sr10r_avg} ± {sr10r_std}")
            print(f"Avg. Spearman's Rho res. (top 20): {sr20r_avg} ± {sr20r_std}")
            print(f"Avg. Spearman's Rho (top 10): {sr10_avg}")
            print(f"Avg. Spearman's Rho (top 20): {sr20_avg}")

        self.eval_metrics = {
            'sr10p': sr10p_avg,
            'sr10p_std': sr10p_std,
            'sr20p': sr20p_avg,
            'sr20p_std': sr20p_std,
            'sr10r': sr10r_avg,
            'sr10r_std': sr10r_std,
            'sr20r': sr20r_avg,
            'sr20r_std': sr20r_std,
            'sr10': sr10_avg,
            'sr20': sr20_avg,
        }

    def predict(self, x: np.ndarray, group_sizes: List[int], y: Optional[np.ndarray] = None,
                stages: Optional[List['CPStage']] = None, riders: Optional[List['CPRider']] = None,
                verbose: Optional[bool] = True) -> Optional[List[CPPrediction]]:
        """
        Predict using the trained model.

        :param x: Input features for prediction.
        :param group_sizes: Sizes of groups in the prediction set.
        :param y: Optional actual output ranks for prediction.
        :param stages: Optional list of stages.
        :param riders: Optional list of riders.
        :param verbose: Whether to print the prediction results.
        :return: List of CPPrediction objects.
        """

        # Make predictions
        predictions = self.model.predict(x)

        # Process predictions per group
        start = 0
        prediction_objs = list()
        for group_size in group_sizes:

            # Define end and retrieve sorted prediction
            end = start + group_size

            # Create prediction
            prediction_obj = CPPrediction(
                prediction=rankdata(-predictions[start:end], method='ordinal'),
                scores=predictions[start:end],
                result=y[start:end] if y is not None else None,
                stage=stages[start] if stages else None,
                riders=riders[start:end] if riders else None,
            )

            # Append prediction object
            prediction_objs.append(prediction_obj)

            # Print prediction object if verbose
            if verbose:
                prediction_obj.print()

            start = end

        return prediction_objs

    def plot(self, feature_names: Tuple[str, ...], savefig: Optional[bool] = False) -> None:
        """
        Plot feature importances of the trained model.

        :param feature_names: Names of the features.
        :param savefig: Whether to save the figure as a file.
        """

        # Retrieve interactions and order
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Setup plot
        plt.figure()
        plt.title("Feature importances model")
        plt.bar(range(len(feature_names)), importances[indices], align="center")
        plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()

        # Optionally save figure
        if savefig:
            plt.savefig(Path('data') / Path(self.dump_fn).with_suffix('.png'))

        # Show plot
        plt.show()

    def dumps(self):
        # Use a temporary directory and file to save the model
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / 'xgb_model.json'
            self.model.save_model(model_path.as_posix())

            # Directly read the model JSON
            with open(model_path, 'r') as f:
                model_json = f.read()

        return {
            "cls": self.__class__.__name__,
            "name": self.name,
            "config": self.config,
            "model_type": self.model_type,
            "model": model_json,
            "eval_metrics": self.eval_metrics,
        }

    @classmethod
    def loads(cls, data: Dict[str, Any]) -> 'XGBModel':
        instance = cls(
            name=data.get('name', None),
            config=data.get('config', {})
        )

        # Use a temporary directory and file to load the model
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / 'xgb_model.json'
            with open(model_path, 'w') as f:
                f.write(data['model'])

            # Directly load the model JSON
            xgb_model = xgb.XGBRanker()
            xgb_model.load_model(model_path.as_posix())

        instance.model = xgb_model
        instance.eval_metrics = data.get('eval_metrics', {})

        return instance

    @staticmethod
    def _convert_ranks_to_relevance(_y_val, _group_sizes, _k=20):
        """
        Convert ranks to relevance scores for NDCG ranking.

        :param _y_val: True values (ranks)
        :param _group_sizes: Sizes of groups in the validation set
        :param _k: Number of top predictions to consider
        :return: Relevance scores for each rank in the validation set
        """
        _y_relevance = np.zeros_like(_y_val, dtype=int)
        _start = 0

        for _group_size in _group_sizes:
            _end = _start + _group_size
            _group_ranks = [_y_val[_i].item() for _i in range(_start, _end)]

            for _rank in _group_ranks:

                # Rank 0, skip (DNF)
                if _rank == 0:
                    pass
                # Assign relevance from 30 downward (i.e. rank 1-20 gets 30, 29, 28, etc. points)
                elif _rank < _k:
                    _index = _start + _group_ranks.index(_rank)
                    _y_relevance[_index] = 30 - (_rank - 1)

            _start = _end

        return _y_relevance

    @staticmethod
    def _spearmanr_k(_y_val, _y_pred, _k=20):
        """
        Calculate Spearman's rank correlation coefficient for the top-k predictions.

        NOTE:
            This function generally compares relevance scores, where higher is better.
            When using ranks (where lower is better), ensure to invert the _y_val input argument (e.g., use -y[...]).

        :param _y_val: True values
        :param _y_pred: Predicted values
        :param _k: Number of top predictions to consider
        :return: Spearman's rank correlation coefficient for the top-k predictions
        """
        # Get indices of top-k ranks
        _k_indices = np.argsort([_yv.item() for _yv in _y_val])[-min(_k, len(_y_val)):]

        # Get actual and predicted ranks for top-k
        _y_val_top_k = [_y_val[_ki] for _ki in _k_indices]
        _y_pred_top_k = [_y_pred[_ki] for _ki in _k_indices]

        return spearmanr(_y_val_top_k, _y_pred_top_k)
