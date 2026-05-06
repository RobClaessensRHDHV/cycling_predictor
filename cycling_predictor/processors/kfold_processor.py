from typing import List

import numpy as np
from sklearn.model_selection import KFold

from cycling_predictor.processors.processor import CPProcessor
from cycling_predictor.dataset import CyclingDataset


# TODO: Make abstract?
class KFoldProcessor(CPProcessor):
    """
    Processor with KFold cross-validation utilities.
    Inherits all batching, feature, and collection logic from CPProcessor.
    """
    # TODO: Use random_state from config?
    def kfold_split(self, n_splits=5, random_state=42) -> List:
        """
        Returns train/val indices for each fold, splitting on batches (stages).
        """
        n_batches = len(self._batches)
        kf = KFold(n_splits=min(n_splits, n_batches), shuffle=True, random_state=random_state)
        return list(kf.split(range(n_batches)))

    def prepare_fold(self, batches: List[CyclingDataset]):
        """
        Flattens batches, returns samples, targets, and group sizes. Assumes data is already scaled.
        """
        samples = np.vstack([batch.samples for batch in batches])
        targets = np.concatenate([batch.targets for batch in batches])
        group_sizes = [len(batch) for batch in batches]
        return samples, targets, group_sizes

    def aggregate_metrics(self, fold_metrics: List[dict]) -> dict:
        """
        Aggregates metrics (mean, std) across folds for each metric key.
        """
        aggregated_metrics = dict()
        for k in fold_metrics[0].keys():
            values = [m.get(k, 0) for m in fold_metrics]
            aggregated_metrics[k] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
        return aggregated_metrics
