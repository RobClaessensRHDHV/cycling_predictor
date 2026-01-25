from typing import Any, Dict, List, Optional
from uuid import uuid4
import json

import numpy as np
from scipy.stats import spearmanr

from cycling_predictor.classes import CPStage, CPRider


class CPPrediction:
    """
    CyclingPredictor Prediction class.
    """
    def __init__(
            self,
            prediction: np.ndarray,
            scores: Optional[np.ndarray] = None,
            result: Optional[np.ndarray] = None,
            stage: Optional[CPStage] = None,
            riders: List[CPRider] = None):
        """
        :param prediction: Prediction, array of integers in predicted order (rank 1 - N).
        :param scores: Optional raw scores from the model (i.e. relevance scores before conversion to ranks).
        :param result: Optional actual result, array of integers in actual order (rank 1 - N).
        :param stage: Predicted stage.
        :param riders: Optional list of riders used in the prediction/result.
        """
        self.prediction = prediction
        self.scores = scores
        self.result = result
        self.stage = stage
        self.riders = riders
        self._uid = str(uuid4())

    def __repr__(self):
        if self.rider_prediction:
            prediction_str = \
                f"{', '.join([f'#{i}. {rider.name}' for i, rider in enumerate(self.rider_prediction[:3], start=1)])}"
        else:
            prediction_str = f"Prediction of {len(self.prediction)} riders"

        if self.stage:
            return f"{self.stage} - {prediction_str}"
        else:
            return prediction_str

    @property
    def uid(self):
        return self._uid

    @property
    def rider_prediction(self) -> List[CPRider]:
        """
        Get the list of predicted riders in order.

        :return: List of predicted riders.
        """
        return [self.riders[i] for i in np.argsort(self.prediction)]

    @property
    def rider_result(self) -> Optional[List[CPRider]]:
        """
        Get the list of actual riders in order.

        :return: List of actual riders.
        """
        if self.result is None:
            return None
        return [self.riders[i] for i in np.argsort(self.result)]

    def print(self, k: Optional[int] = 20):
        """
        Print the prediction.

        :param k: Number of top riders to print.
        """

        print(f"\n{self.stage.name} {self.stage.year} "
              f"{f'- Stage {self.stage.stage_number}' if self.stage.stage_number else ''} "
              f"({self.stage.stage_type}{self.stage.stage_profile}):")

        # If riders are available, print top-k predicted riders
        if self.rider_prediction:
            for rank, rider in enumerate(self.rider_prediction[:k] or list(), start=1):

                # If result is available, print actual rank between parentheses
                if self.result is not None:
                    print(f"#{rank}: {rider.name}".ljust(40) +
                          f"(#{self.rider_result.index(rider) + 1 if rider in self.rider_result else 'N/A'})")
                else:
                    print(f"#{rank}: {rider.name}")

        # If result is available, print Spearman's rank correlation coefficient
        if self.result is not None:
            print(f"Spearman's Rho pred. ({k}):".ljust(40) + f"{self.spearmanr_k_prediction(k):.3f}")
            print(f"Spearman's Rho res. ({k}):".ljust(40) + f"{self.spearmanr_k_result(k):.3f}")

    def spearmanr_k_prediction(self, k: Optional[int] = 20) -> Optional[float]:
        """
        Calculate Spearman's rank correlation coefficient for the top-k predicted riders.
        This is a measure on how well riders that are predicted best are ranked by the model.

        :param k: Number of top predictions to consider.
        :return: Spearman's rank correlation coefficient for the top-k predictions.
        """

        if self.result is None:
            print(f"Result for stage {self.stage} not available, Spearman's rank correlation coefficient not computed.")
            return None

        # Get predicted and actual ranks for predicted top-k riders
        y_pre, y_res = list(), list()
        for i in np.argsort(self.prediction)[:k]:
            y_pre.append(self.prediction[i])
            y_res.append(self.result[i])

        # Compute Spearman's rank correlation coefficient for predicted top-k riders
        return spearmanr(y_pre, y_res).correlation

    def spearmanr_k_result(self, k: Optional[int] = 20) -> Optional[float]:
        """
        Calculate Spearman's rank correlation coefficient for the top-k actual riders.
        This is a measure on how well the riders that actually performed best are ranked by the model.

        :param k: Number of top results to consider.
        :return: Spearman's rank correlation coefficient for the top-k results.
        """

        if self.result is None:
            print(f"Result for stage {self.stage} not available, Spearman's rank correlation coefficient not computed.")
            return None

        # Get predicted and actual ranks for actual top-k riders
        y_pre, y_res = list(), list()
        for i in np.argsort(self.result)[:k]:
            y_pre.append(self.prediction[i])
            y_res.append(self.result[i])

        # Compute Spearman's rank correlation coefficient for actual top-k riders
        return spearmanr(y_pre, y_res).correlation

    def dumps(self):
        return {
            "cls": self.__class__.__name__,
            "prediction": self.prediction.tolist(),
            "scores": self.scores.tolist() if self.scores is not None else None,
            "result": self.result.tolist() if self.result is not None else None,
            "stage": self.stage.dumps() if self.stage else None,
            "riders": [rider.dumps() for rider in self.riders],
        }

    @classmethod
    def loads(cls, data: Dict[str, Any]) -> 'CPPrediction':
        riders = [CPRider.loads(rider) for rider in data.get("riders", [])]
        prediction = cls(
            prediction=np.array(data["prediction"]),
            scores=np.array(data["scores"]) if data.get("scores") is not None else None,
            result=np.array(data["result"]) if data.get("result") is not None else None,
            stage=CPStage.loads(data["stage"]) if data.get("stage") else None,
            riders=riders,
        )
        prediction._uid = data.get("uid", str(uuid4()))
        return prediction

    def dump(self, fp: Optional[str] = None):
        data = self.dumps()
        if not fp:
            if self.stage:
                if self.stage.stage_number:
                    fp = f'data/prediction_{self.stage.name}_stage{self.stage.stage_number}_{self.stage.year}.json'
                else:
                    fp = f'data/prediction_{self.stage.name}_{self.stage.year}.json'
            else:
                fp = f'data/prediction_{self.uid}.json'

        with open(fp, 'w+') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, fp: str) -> 'CPPrediction':
        with open(fp, 'r') as f:
            data = json.load(f)
        return cls.loads(data)
