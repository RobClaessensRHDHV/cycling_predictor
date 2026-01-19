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
                f"{'_'.join([v for val in self.stage_filter.values() for v in val]) if self.stage_filter else list()}.json")

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
        Train the model using the preprocessed data.

        :param verbose: Whether to print training results.
        """

        if self.dataloader is None:
            raise ValueError("DataLoader is not initialized. Please run preprocess() first.")
        if self.model is None:
            raise ValueError("Model is not initialized. Please assign a model first.")

        # Retrieve data (assuming one batch?)
        data = next(iter(self.dataloader))
        samples = data['samples']
        targets = data['targets']
        stages = data['stages']

        # Get set of stages and train_test_split
        unique_stages = list(dict.fromkeys(stages))
        unique_train_stages, unique_test_stages = train_test_split(
            unique_stages,
            test_size=self.config.get('test_size'),
            random_state=self.config.get('random_state'))

        # Collect samples and targets
        train_samples, train_targets, train_stages, test_samples, test_targets, test_stages = \
            list(), list(), list(), list(), list(), list()
        for sample, target, stage in zip(samples, targets, stages):
            if stage in unique_train_stages:
                train_samples.append(sample.numpy())
                train_targets.append(target.item())
                train_stages.append(stage)
            else:
                test_samples.append(sample.numpy())
                test_targets.append(target.item())
                test_stages.append(stage)

        # Sort samples, targets and stages, based on stages
        train_stages = [self.collector.get_stage(s) or s for s in train_stages]
        training_data = list(zip(train_samples, train_targets, train_stages))
        training_data.sort(key=lambda x: getattr(x[2], 'start_date', x[2]))
        train_samples, train_targets, train_stages = zip(*training_data)

        test_stages = [self.collector.get_stage(s) or s for s in test_stages]
        testing_data = list(zip(test_samples, test_targets, test_stages))
        testing_data.sort(key=lambda x: getattr(x[2], 'start_date', x[2]))
        test_samples, test_targets, test_stages = zip(*testing_data)

        # Convert to numpy arrays
        x_train = np.vstack(train_samples)
        y_train = np.array(train_targets)
        train_group_sizes = list()
        for i, stage in enumerate(train_stages):
            if i == 0 or stage != train_stages[i - 1]:
                train_group_sizes.append(train_stages.count(stage))

        x_test = np.vstack(test_samples)
        y_test = np.array(test_targets)
        test_group_sizes = list()
        for i, stage in enumerate(test_stages):
            if i == 0 or stage != test_stages[i - 1]:
                test_group_sizes.append(test_stages.count(stage))

        # Train the model
        self.model.train(x_train, train_group_sizes, y_train, verbose=verbose)

        # Test the model
        self.model.test(x_test, test_group_sizes, y_test, verbose=verbose)


if __name__ == "__main__":

    # Get entry collector
    _entry_collector = CPGTEntryCollector.load(r'..\collectors\data\entry_collector_giro_tour_vuelta_2023_2024_2025_100.json')

    # Set up trainer (RR1)
    trainer = CPTrainer(
        collector=_entry_collector,
        rider_feature_filter=('pr_', 'tts', 'ttl', 'cob', 'mtn', 'gc_'),
        stage_feature_filter=('race_startlist_quality_score',),
        interactions={
            ('spr', 'gradient_final_km'): op.sub,
        },
        stage_filter={'stage_profile': (1,), 'stage_type': ('RR',)},
    )

    # # Set up trainer (RR4 & RR5)
    # trainer = CPTrainer(
    #     collector=_entry_collector,
    #     rider_feature_filter=('pr_', 'tts', 'ttl', 'flt', 'spr', 'cob'),
    #     stage_feature_filter=('race_startlist_quality_score',),
    #     interactions={
    #         ('hll', 'profile_score'): op.add,
    #         ('mtn', 'vertical_meters'): op.add,
    #     },
    #     stage_filter={'stage_profile': (4, 5,), 'stage_type': ('RR',)},
    # )

    # Preprocess data
    trainer.preprocess()

    # Initialize model
    xgb_model = XGBModel(
        config={
            'k': 10,
            'learning_rate': 0.01,
            'max_depth': 10,
            'alpha': 1,
            'n_estimators': 500,
        }
    )

    # Set model
    trainer.model = xgb_model

    # Train model
    random_state = 15     # Best RR1
    # random_state = 4      # Best RR45
    # random_state = range(1, 51)
    if isinstance(random_state, int):
        trainer.config = {'random_state': random_state}
        trainer.train()
        trainer.plot()
    else:
        sr20s = list()
        for i in random_state:
            trainer.config = {'random_state': i}
            trainer.train()
            # trainer.plot()

            sr20s.append(trainer.model.eval_metrics.get('sr20', 0))

        print(f"Avg SRK20 over 100 runs: {np.mean(sr20s):.3f} ± {np.std(sr20s):.3f}")
        print(f"Max SRK20 over 100 runs: {max(sr20s):.3f}")

    # Dump trainer
    trainer.dump()
