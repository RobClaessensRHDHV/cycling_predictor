from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import json
import operator as op

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from cycling_predictor.classes import CPEntry
from cycling_predictor.collectors import CPEntryCollector
from cycling_predictor.dataset import CyclingDataset
from cycling_predictor.models import *


class CPProcessor(ABC):
    """
    CyclingProcessor Processor class.
    """
    def __init__(
            self,
            collector: CPEntryCollector,
            rider_feature_filter: Optional[Tuple[str, ...]] = None,
            stage_feature_filter: Optional[Tuple[str, ...]] = None,
            interactions: Optional[Dict[Tuple[str, ...], op]] = None,
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
        :param model: Model to use.
        :param config: Configuration of the predictor.
        """
        self.collector = collector
        self.rider_feature_filter = rider_feature_filter
        self.stage_feature_filter = stage_feature_filter
        self.interactions = interactions
        self.rider_filter = rider_filter
        self.stage_filter = stage_filter
        self.scaler = scaler
        self.model = model
        self.config = config
        self.dataloader: Optional[DataLoader] = None

    @property
    def rider_feature_filter(self) -> Optional[Tuple[str, ...]]:
        try:
            return self.__rider_feature_filter
        except AttributeError:
            return None

    @rider_feature_filter.setter
    def rider_feature_filter(self, rider_feature_filter: Optional[Tuple[str, ...]] = None) -> None:
        self.__rider_feature_filter = rider_feature_filter
        self._validate_interactions()

    @property
    def stage_feature_filter(self) -> Optional[Tuple[str, ...]]:
        try:
            return self.__stage_feature_filter
        except AttributeError:
            return None

    @stage_feature_filter.setter
    def stage_feature_filter(self, stage_feature_filter: Optional[Tuple[str, ...]] = None) -> None:
        self.__stage_feature_filter = stage_feature_filter
        self._validate_interactions()

    @property
    def interactions(self) -> Optional[Dict[Tuple[str, str], op]]:
        try:
            return self.__interactions
        except AttributeError:
            return None

    @interactions.setter
    def interactions(self, interactions: Optional[Dict[Tuple[str, str], op]] = None) -> None:
        self.__interactions = interactions
        self._validate_interactions()

    def _validate_interactions(self):
        features = [fn for fn in (CPEntry._rider_sample_keys + CPEntry._stage_sample_keys + CPEntry._entry_sample_keys)
                    if fn not in ((self.rider_feature_filter or tuple()) + (self.stage_feature_filter or tuple()))]

        # Validate interactions
        if self.interactions:
            for (f1, f2), operation in self.interactions.items():
                if f1 not in features or f2 not in features:
                    raise ValueError(f"Features used for interaction ({f1} and/or {f2}) not in available features.")

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            'MIN_DENOMINATOR': {
                'gradient_final_km': 0.1,
            }
        }

    @property
    def config(self) -> Dict[str, Any]:
        return self.__config

    @config.setter
    def config(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not config:
            self.__config = self.default_config
        else:
            for key, value in self.default_config.items():
                if key not in config:
                    config[key] = value
            self.__config = config

    @property
    def feature_names(self) -> Tuple[str, ...]:
        features = [fn for fn in (CPEntry._rider_sample_keys + CPEntry._stage_sample_keys + CPEntry._entry_sample_keys)
                    if fn not in ((self.rider_feature_filter or tuple()) + (self.stage_feature_filter or tuple()))]

        for (f1, f2), operation in self.interactions.items():
            features.append(f"{f1}_{operation.__name__}_{f2}")

        return tuple(features)

    @property
    @abstractmethod
    def dump_fn(self) -> str:
        return f"{self.__class__.__name__}.json"

    @abstractmethod
    def scale(self, samples: np.ndarray) -> np.ndarray:
        """
        Scale samples using the provided scaler.

        :param samples: Samples to scale.
        :return: Scaled samples.
        """
        pass

    def preprocess(self, batch_size: Optional[int] = 0, rider_feature_noise: Optional[float] = None) -> None:
        """
        Preprocess data from the entry collector and create a DataLoader.

        :param batch_size: Optional batch size for the DataLoader. If 0 or None, use the entire dataset as a single batch.
        :param rider_feature_noise: Optional noise to add to rider features for augmentation.
        :return:
        """
        samples, targets, stages, riders = list(), list(), list(), list()

        for stage in self.collector.stages:
            skip_stage = False
            if self.stage_filter:
                for key, value in self.stage_filter.items():
                    attr = getattr(stage, key)
                    if isinstance(attr, (list, tuple, set)):
                        if not any([a in value for a in attr]):
                            print(f"Skipping stage {stage} due to stage filter {value} on {key}.")
                            skip_stage = True
                            break
                    else:
                        if attr not in value:
                            print(f"Skipping stage {stage} due to stage filter {value} on {key}.")
                            skip_stage = True
                            break

            # Skip stage if flagged
            if skip_stage:
                continue

            entries = self.collector.get_entries_per_stage(stage)
            for entry in entries:
                sample, target, stage_uid, rider_uid = entry.to_data(self.rider_feature_filter, self.stage_feature_filter)
                samples.append(sample)
                targets.append(target)
                stages.append(stage_uid)
                riders.append(rider_uid)

        # TODO: Possibly also do interactions after normalization
        # Include interactions in samples, applied on numpy arrays
        samples = np.array(samples)
        if self.interactions:
            for (f1, f2), operation in self.interactions.items():
                if f1 in self.feature_names and f2 in self.feature_names:
                    idx1 = self.feature_names.index(f1)
                    idx2 = self.feature_names.index(f2)
                    if operation == op.truediv and f2 in self.config.get('MIN_DENOMINATOR', {}):
                        interaction_feature = operation(
                            samples[:, idx1], np.clip(samples[:, idx2], self.config.get('MIN_DENOMINATOR')[f2], None)
                        ).reshape(-1, 1)
                    else:
                        interaction_feature = operation(samples[:, idx1], samples[:, idx2]).reshape(-1, 1)
                    samples = np.hstack((samples, interaction_feature))
                    print(f"Added interaction feature: {f1} {operation.__name__} {f2}")
                else:
                    print(f"Skipping interaction feature for {f1} and {f2} as one or both features are missing.")

        # Normalize samples
        samples = self.scale(samples)

        # Slightly augment rider features of samples with some noise in order to prevent (or test) overfitting on riders
        # NOTE: Not recommended to apply in general, but useful for experimentation / validation
        if rider_feature_noise:
            for i, fn in enumerate(self.feature_names):
                # TODO: Probably include 'form' features in augmentation
                if fn in CPEntry._rider_sample_keys:
                    samples[:, i] += np.random.normal(scale=rider_feature_noise, size=samples.shape[0])

        # Create DataLoader
        dataset = CyclingDataset(samples, targets, stages, riders)
        self.dataloader = DataLoader(dataset, batch_size=batch_size or len(dataset))

        print(f"Created dataloader with {len(dataset)} samples of {len(set(stages))} stages and {len(set(riders))} riders.")

    def plot(self):
        """
        Plot evaluation metrics of the trained model.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please assign a model first.")

        self.model.plot(self.feature_names)

    def dumps(self):
        return {
            "cls": self.__class__.__name__,
            "collector": self.collector.dumps(),
            "rider_feature_filter": self.rider_feature_filter,
            "stage_feature_filter": self.stage_feature_filter,
            "interactions": {
                '-'.join(k): v.__name__ for k, v in self.interactions.items()
            },
            "rider_filter": self.rider_filter,
            "stage_filter": self.stage_filter,
            "scaler": {
                "mean": self.scaler.mean_.tolist() if self.scaler else None,
                "scale": self.scaler.scale_.tolist() if self.scaler else None,
                "var": self.scaler.var_.tolist() if self.scaler else None,
            },
            "model": self.model.dumps() if self.model else None,
            "config": self.config,

        }

    @classmethod
    def loads(cls, data: Dict[str, Any], sub_cls: Optional[type] = None) -> 'CPProcessor':
        # Retrieve subclass if not provided
        if not sub_cls:
            for sub_cls in cls.__subclasses__():
                if sub_cls.__name__ == data.get("cls"):
                    break
            else:
                sub_cls = cls

        # Create instance
        instance = sub_cls(
            collector=CPEntryCollector.loads(data['collector']),
            rider_feature_filter=tuple(data['rider_feature_filter']) if data['rider_feature_filter'] else None,
            stage_feature_filter=tuple(data['stage_feature_filter']) if data['stage_feature_filter'] else None,
            interactions={tuple(k.split('-')): getattr(op, v) for k, v in data['interactions'].items()},
            rider_filter=data['rider_filter'],
            stage_filter=data['stage_filter'],
            model=BaseModel.loads(data['model']) if data['model'] else None,
            config=data['config'],
        )

        # Assign scaler
        if data['scaler']:
            scaler = StandardScaler()
            scaler.mean_ = np.array(data['scaler']['mean'])
            scaler.scale_ =  np.array(data['scaler']['scale'])
            scaler.var_ =  np.array(data['scaler']['var'])
            instance.scaler = scaler

        return instance

    def dump(self, fp: Optional[str] = None):
        data = self.dumps()
        if not fp:
            fp = Path(__file__).parent / 'data' / self.dump_fn

        with open(fp, 'w+') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, fp: str) -> 'CPProcessor':
        with open(fp, 'r') as f:
            data = json.load(f)
        return cls.loads(data)