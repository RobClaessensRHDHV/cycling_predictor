from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json

import numpy as np


class BaseModel(ABC):
    """
    CyclingPredictor BaseModel class.
    """
    @abstractmethod
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.name = name if name else self.__class__.__name__
        self.config = config
        self.model = None
        self.eval_metrics = None

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    @property
    @abstractmethod
    def default_config(self) -> Dict[str, Any]:
        pass

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
    def eval_metrics(self) -> Dict[str, Any]:
        return self.__eval_metrics

    @eval_metrics.setter
    def eval_metrics(self, eval_metrics: Optional[Dict[str, Any]]) -> None:
        if isinstance(eval_metrics, dict):
            self.__eval_metrics = eval_metrics
        elif eval_metrics is None:
            self.__eval_metrics = dict()
        else:
            raise TypeError("eval_metrics must be a dictionary or None.")

    @property
    @abstractmethod
    def dump_fn(self) -> str:
        return f"{self.name}.json"

    @abstractmethod
    def train(self, x: np.ndarray, group_sizes: List[int], y: np.ndarray, verbose: Optional[bool] = True) -> None:
        pass

    @abstractmethod
    def test(self, x: np.ndarray, group_sizes: List[int], y: np.ndarray, verbose: Optional[bool] = True) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray, group_sizes: List[int], y: Optional[np.ndarray] = None,
                stages: Optional[List['CPStage']] = None, riders: Optional[List['CPRider']] = None,
                verbose: Optional[bool] = True) -> Optional[List['CPPrediction']]:
        pass

    @abstractmethod
    def plot(self, feature_names: Tuple[str, ...], savefig: Optional[bool] = False) -> None:
        pass

    @abstractmethod
    def dumps(self):
        pass

    @classmethod
    @abstractmethod
    def loads(cls, data: Dict[str, Any]) -> 'BaseModel':
        for sub_cls in cls.__subclasses__():
            if sub_cls.__name__ == data.get("cls"):
                return sub_cls.loads(data)
        else:
            raise ValueError(f"Unknown BaseModel subclass: {data.get('cls')}")

    def dump(self, fp: Optional[str] = None):
        data = self.dumps()
        if not fp:
            fp = Path(__file__).parent / 'data' / self.dump_fn

        with open(fp, 'w+') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, fp: str) -> 'BaseModel':
        with open(fp, 'r') as f:
            data = json.load(f)
        return cls.loads(data)
