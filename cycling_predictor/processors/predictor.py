from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import operator as op

import numpy as np
from sklearn.preprocessing import StandardScaler

from cycling_predictor.collectors import CPEntryCollector
from cycling_predictor.models import *
from cycling_predictor.processors import CPProcessor


class CPPredictor(CPProcessor):
    """
    CyclingPredictor Predictor class.
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
        :param model: Model to use for prediction.
        :param config: Configuration of the predictor.
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
    def dump_fn(self) -> str:
        collector_fn = self.collector.dump_fn.split('_', 1)[1]
        return (f"{self.__class__.__name__}_{Path(collector_fn).stem}_"
                f"{'_'.join([str(v) for val in self.stage_filter.values() for v in val]) if self.stage_filter else list()}_update.json")

    def scale(self, samples: np.ndarray) -> np.ndarray:
        """
        Scale the samples using the scaler.

        :param samples: Samples to scale.
        :return: Scaled samples.
        """

        if self.scaler is None:
            raise ValueError("Scaler is not initialized. Please assign a scaler first.")

        return self.scaler.transform(samples)

    def predict(self, verbose: Optional[bool] = True):
        """
        Predict with the model using the preprocessed data.

        :param verbose: Whether to print the training results.
        """

        if self.dataloader is None:
            raise ValueError("DataLoader is not initialized. Please run preprocess() first.")
        if self.model is None:
            raise ValueError("Model is not initialized. Please assign a model first.")

        # Retrieve data (assuming one batch?)
        data = next(iter(self.dataloader))
        samples = data['samples']
        targets = data['targets']
        stages = [self.collector.get_stage(s) or s for s in data['stages']]
        riders = [self.collector.get_rider(r) or r for r in data['riders']]

        # Sort samples, targets and stages, based on stages
        sorted_data = list(zip(samples.numpy(), targets.numpy(), stages, riders))
        sorted_data.sort(key=lambda x: getattr(x[2], 'start_date', x[2]))
        samples, targets, stages, riders = zip(*sorted_data)

        # Convert to numpy arrays
        samples = np.vstack(samples)
        targets = np.array(targets)
        group_sizes = list()
        for i, stage in enumerate(stages):
            if i == 0 or stage != stages[i - 1]:
                group_sizes.append(stages.count(stage))

        # Predict per stage, using the model
        predictions = self.model.predict(samples, group_sizes, targets, stages, riders, verbose)
        return predictions


if __name__ == "__main__":

    from cycling_predictor.collectors import CPEntryCollector
    from cycling_predictor.processors import CPTrainer

    # Load collector
    _collector = CPEntryCollector.load(r'..\collectors\data\CPClassicEntryCollector_classics_2026_update.json')
    # _collector = CPEntryCollector.load(r'..\collectors\data\CPGTEntryCollector_paris_nice_tirreno_adriatico_2026.json')

    # Load trainer
    # _trainer = CPTrainer.load(r'data\CPTrainer_classics_2023_2024_2025_50_0.2_20_RR_sprint.json')
    # _trainer = CPTrainer.load(r'data\CPTrainer_classics_2023_2024_2025_50_0.2_15_RR_cobbles.json')
    _trainer = CPTrainer.load(r'data\CPTrainer_classics_2023_2024_2025_50_0.2_29_RR_hills.json')
    # _trainer = CPTrainer.load(r'data\CPTrainer_paris_nice_tirreno_adriatico_2023_2024_2025_50_0.2_32_RR_1.json')
    # _trainer = CPTrainer.load(r'data\CPTrainer_paris_nice_tirreno_adriatico_2023_2024_2025_50_0.2_11_RR_5.json')

    # Set up predictor with trained model
    _predictor = CPPredictor(
        collector=_collector,
        rider_feature_filter=_trainer.rider_feature_filter,
        stage_feature_filter=_trainer.stage_feature_filter,
        interactions=_trainer.interactions,
        stage_filter=_trainer.stage_filter,
        scaler=_trainer.scaler,
        model=_trainer.model,
    )

    # Preprocess data for prediction
    _predictor.preprocess()

    # Predict
    predictions = _predictor.predict()

    # Dump predictor
    _predictor.dump()
