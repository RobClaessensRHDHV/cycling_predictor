from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

from cycling_predictor.predictions import CPPrediction
from cycling_predictor.processors import CPPredictor


class CPEnsemblePredictor:
    """
    CyclingPredictor EnsemblePredictor class. Acts as an ensemble for multiple specialized CPPredictors, allowing to
    predict e.g. a range of classics, or a whole grand tour, combining specialized models for different stage types.
    """
    def __init__(self, predictors: List[CPPredictor]):
        """
        :param predictors: List of specialized predictors to ensemble.
        """
        self.predictors = predictors

    def preprocess(self, **kwargs) -> None:
        """
        Delegates preprocessing to all child predictors.
        """
        for predictor in self.predictors:
            predictor.preprocess(**kwargs)

    def predict(self) -> List[CPPrediction]:
        """
        Collects results from all child predictors and merges them.
        """
        # Collect all predictions from children
        prediction_dict: Dict[str, List[CPPrediction]] = dict()
        
        for predictor in self.predictors:
            child_predictions = predictor.predict()
            
            for child_prediction in child_predictions:
                if child_prediction.stage:
                    stage_uid = child_prediction.stage.uid
                    if prediction_dict.get(stage_uid):
                        prediction_dict[stage_uid].append(child_prediction)
                    else:
                        prediction_dict[stage_uid] = [child_predictions]

        # Iterate through collected stages and finalize
        final_predictions = []
        for stage_uid, stage_predictions in prediction_dict.items():
            # If only one prediction, directly pass along
            if len(stage_predictions) == 1:
                final_predictions.append(stage_predictions[0])
            # Otherwise, combine predictions
            else:
                final_predictions.append(self._combine_predictions(stage_predictions))

        return final_predictions

    @staticmethod
    def _combine_predictions(predictions: List[CPPrediction]) -> CPPrediction:
        """
        Combines multiple predictions for the same stage.
        If models provide raw scores, it uses normalized relevance scores.
        Otherwise, it falls back to rank-based fusion.
        """
        reference_prediction = predictions[0]
        num_riders = len(reference_prediction.prediction)
        
        # Calculate a consensus score where HIGHER is BETTER
        consensus_scores = np.zeros(num_riders)
        num_predictors = len(predictions)

        for prediction in predictions:
            if prediction.scores is not None:
                # Use raw scores. First, normalize to [0, 1] within this specific prediction
                # Note: XGBRanker scores are usually such that higher is better relevance.
                s_min, s_max = prediction.scores.min(), prediction.scores.max()
                if s_max > s_min:
                    normalized_scores = (prediction.scores - s_min) / (s_max - s_min)
                # If all scores are identical, treat all as equally relevant
                else:
                    normalized_scores = np.ones(num_riders)
                consensus_scores += normalized_scores / num_predictors
            else:
                # Fallback to rank-based. Turn ranks (1=best) into scores (higher=best)
                # Max rank - current rank gives a simple inverted scale
                rank_scores = (num_riders - prediction.prediction + 1) / num_riders
                consensus_scores += rank_scores / num_predictors

        # Re-rank based on consensus scores (descending)
        combined_ranking = rankdata(-consensus_scores, method='ordinal')

        # Return combined prediction, storing consensus scores as well
        return CPPrediction(
            prediction=combined_ranking,
            scores=consensus_scores,
            result=reference_prediction.result,
            stage=reference_prediction.stage,
            riders=reference_prediction.riders
        )
