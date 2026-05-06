from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import operator as op

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from cycling_predictor.collectors import CPEntryCollector, CPGTEntryCollector
from cycling_predictor.models import *
from cycling_predictor.processors.processor import CPProcessor


class CPTrainer(CPProcessor):
    """
    CyclingPredictor Trainer class.
    """
    def __init__(
            self,
            collector: CPEntryCollector,
            rider_feature_filter: Optional[Tuple[str, ...]] = None,
            stage_feature_filter: Optional[Tuple[str, ...]] = None,
            entry_feature_filter: Optional[Tuple[str, ...]] = None,
            interactions: Optional[Dict[Tuple[str, str], op]] = None,
            rider_filter: Optional[Dict[str, Any]] = None,
            stage_filter: Optional[Dict[str, Any]] = None,
            scaler: Optional[StandardScaler] = None,
            model: Optional[BaseModel] = None,
            config: Optional[Dict[str, Any]] = None):
        """
        :param collector: Entry collector from which data will be retrieved.
        :param rider_feature_filter: Tuple of rider features to exclude.
        :param stage_feature_filter: Tuple of stage features to exclude.
        :param entry_feature_filter: Tuple of entry features to exclude.
        :param interactions: Interaction features to create.
        :param rider_filter: Filter for riders to include.
        :param stage_filter: Filter for stages to include.
        :param scaler: Scaler for normalizing features.
        :param model: Model to train.
        :param config: Configuration of the trainer.
        """
        super().__init__(
            collector=collector,
            rider_feature_filter=rider_feature_filter,
            stage_feature_filter=stage_feature_filter,
            entry_feature_filter=entry_feature_filter,
            interactions=interactions,
            rider_filter=rider_filter,
            stage_filter=stage_filter,
            scaler=scaler,
            model=model,
            config=config,
        )

    @property
    def default_config(self) -> Dict[str, Any]:
        return super().default_config | {
            'test_size': 0.2,
            'random_state': 42
        }

    @property
    def dump_fn(self) -> str:
        collector_fn = self.collector.dump_fn.split('_',1)[1]
        return (f"{self.__class__.__name__}_{Path(collector_fn).stem}_"
                f"{self.config.get('test_size')}_{self.config.get('random_state')}_"
                f"{'_'.join([str(v) for val in self.stage_filter.values() for v in val]) if self.stage_filter else list()}.json")

    def scale(self, samples: np.ndarray) ->np.ndarray:
        """
        Scale the samples using the scaler.

        :param samples: Samples to scale.
        :return: Scaled samples.
        """
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(samples)
        else:
            return self.scaler.transform(samples)

    def train(self, verbose: bool = True):
        """
        Train the model using the preprocessed, batched data.

        :param verbose: Whether to print training results.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please assign a model first.")

        # Preprocess data
        self.preprocess()

        # Get set of stages and train_test_split
        unique_stages = list(dict.fromkeys(self._stages))
        unique_train_stages, unique_test_stages = train_test_split(
            unique_stages,
            test_size=self.config.get('test_size'),
            random_state=self.config.get('random_state'))

        # Assign batches to train/test
        train_batches = [batch for batch in self._batches if batch.stages[0] in unique_train_stages]
        test_batches = [batch for batch in self._batches if batch.stages[0] in unique_test_stages]

        # Flatten train/test samples and targets
        x_train = np.vstack([batch.samples for batch in train_batches])
        y_train = np.concatenate([batch.targets for batch in train_batches])
        train_group_sizes = [len(batch.targets) for batch in train_batches]

        x_test = np.vstack([batch.samples for batch in test_batches])
        y_test = np.concatenate([batch.targets for batch in test_batches])
        test_group_sizes = [len(batch.targets) for batch in test_batches]

        # Train the model
        self.model.train(x_train, train_group_sizes, y_train, verbose=verbose)

        # Test the model
        self.model.test(x_test, test_group_sizes, y_test, verbose=verbose)
