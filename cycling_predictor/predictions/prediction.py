from typing import Any, Dict, List, Optional
from uuid import uuid4
import json

import numpy as np
from scipy.stats import spearmanr

from cycling_predictor.classes import CPStage, CPRider
from cycling_predictor.maps import CPCOPointsMap, CPCOFactorMap


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
                if self.result is not None and any(self.result):
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

    def co_score(self) -> Optional[float]:
        """
        Compute the quality score for the prediction.
        A rider in the top-10 of the prediction generates points if they finish in the top-20.
        Points are multiplied by a factor depending on the predicted rank.

        :return: Quality score for the prediction.
        """
        if self.result is None:
            return None

        score = 0.0

        # Get indices of top 10 predicted riders (rankings are 1-based, prediction indices are 0-based)
        # prediction array contains ranks (1 to N) for each rider index.
        # So np.argsort(self.prediction) gives indices of riders in order of their predicted rank.
        top_10_pred_indices = np.argsort(self.prediction)[:10]

        for i, rider_idx in enumerate(top_10_pred_indices):

            predicted_rank = i + 1

            # Get actual rank for this rider
            # result array contains ranks (1 to N) for each rider index.
            actual_rank = self.result[rider_idx]

            if actual_rank in CPCOPointsMap:
                factor = CPCOFactorMap.get(predicted_rank, 1.0)
                points = CPCOPointsMap[actual_rank]
                score += points * factor
            else:
                factor = 0
                points = 0

            print(f"{self.riders[int(rider_idx)].name} - "
                  f"pred. {predicted_rank} ({factor}x) - "
                  f"res. {actual_rank} ({points}): "
                  f"{points * factor:.3f}")

        return score

    def dumps(self):
        return {
            "cls": self.__class__.__name__,
            "prediction": self.prediction.tolist(),
            "scores": self.scores.tolist() if self.scores is not None else None,
            "result": self.result.tolist() if self.result is not None else None,
            "stage": self.stage.dumps() if self.stage else None,
            "riders": [rider.dumps() for rider in self.riders],
            "rider_prediction": {i: rider.name for i, rider in enumerate(self.rider_prediction, start=1)}
                if self.rider_prediction else None,
            "rider_result": {i: rider.name for i, rider in enumerate(self.rider_result, start=1)}
                if self.rider_result else None,
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
                    fp = f'data/prediction_{self.stage.name.replace('-', '_')}_stage{self.stage.stage_number}_{self.stage.year}.json'
                else:
                    fp = f'data/prediction_{self.stage.name.replace('-', '_')}_{self.stage.year}.json'
            else:
                fp = f'data/prediction_{self.uid}.json'

        with open(fp, 'w+') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, fp: str) -> 'CPPrediction':
        with open(fp, 'r') as f:
            data = json.load(f)
        return cls.loads(data)
