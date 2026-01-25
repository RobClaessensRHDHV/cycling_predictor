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

    def predict(self, normalize: bool = False) -> List[CPPrediction]:
        """
        Collects results from all child predictors and merges them.

        :param normalize: Whether to normalize raw scores when combining predictions.
        :return: List of combined predictions.
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
                        prediction_dict[stage_uid] = [child_prediction]

        # Iterate through collected stages and finalize
        predictions = list()
        for stage_uid, stage_predictions in prediction_dict.items():
            # If only one prediction, directly pass along
            if len(stage_predictions) == 1:
                predictions.append(stage_predictions[0])
            # Otherwise, combine predictions
            else:
                predictions.append(self._combine_predictions(stage_predictions, normalize=normalize))

        return predictions

    @staticmethod
    def _combine_predictions(predictions: List[CPPrediction], normalize: bool = False) -> CPPrediction:
        """
        Combines multiple predictions for the same stage.
        If models provide raw scores, it uses relevance scores.
        Otherwise, it falls back to rank-based fusion.

        :param predictions: List of predictions to combine.
        :param normalize: Whether to normalize raw scores when combining predictions.
        :return: Combined prediction.
        """
        reference_prediction = predictions[0]
        num_riders = len(reference_prediction.prediction)
        
        # Calculate a consensus score (where higher is better!)
        consensus_scores = np.zeros(num_riders)
        num_predictors = len(predictions)

        for prediction in predictions:
            if prediction.scores is not None:
                # Use raw scores (optionally normalized)
                scores = prediction.scores
                if normalize:
                    s_min, s_max = scores.min(), scores.max()
                    scores = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.ones(num_riders)
                consensus_scores += scores / num_predictors
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


if __name__ == "__main__":

    # Load predictors
    _sprint_predictor = CPPredictor.load(r'data\CPPredictor_classics_2022_50_2022_sprint.json')
    _cobbles_predictor = CPPredictor.load(r'data\CPPredictor_classics_2022_50_2022_cobbles.json')
    _hills_predictor = CPPredictor.load(r'data\CPPredictor_classics_2022_50_2022_hills.json')

    # Set up ensemble predictor
    _ensemble_predictor = CPEnsemblePredictor(
        predictors=[
            _sprint_predictor,
            _cobbles_predictor,
            _hills_predictor
        ]
    )

    # Preprocess data for predictions
    _ensemble_predictor.preprocess()

    # Predict
    _ensemble_predictor.predict()
