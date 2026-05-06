from typing import Optional, Dict, Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from cycling_predictor.processors.kfold_processor import KFoldProcessor
from cycling_predictor.processors.trainer import CPTrainer
from cycling_predictor.models import BaseModel
from cycling_predictor.collectors import CPEntryCollector


class KFoldTrainer(KFoldProcessor):
    """
    Trainer for KFold cross-validation model training and ensembling.
    Each fold is a CPTrainer with its own model and scaler.
    """
    def __init__(
            self,
            collector: CPEntryCollector,
            rider_feature_filter: Optional[tuple] = None,
            stage_feature_filter: Optional[tuple] = None,
            entry_feature_filter: Optional[tuple] = None,
            interactions: Optional[Dict] = None,
            rider_filter: Optional[Dict[str, Any]] = None,
            stage_filter: Optional[Dict[str, Any]] = None,
            scaler: Optional[Any] = None,
            model: Optional[BaseModel] = None,
            config: Optional[Dict[str, Any]] = None):
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
        self.trainers = []

    @property
    def dump_fn(self) -> str:
        # TODO: Update dump_fn
        collector_fn = self.collector.dump_fn.split('_', 1)[1]
        return (f"{self.__class__.__name__}_{collector_fn.split('.')[0]}_"
                f"{self.config.get('test_size', 'NA')}_{self.config.get('random_state', 'NA')}_"
                f"{self.model.__class__.__name__}.json")

    def scale(self, samples: np.ndarray) ->np.ndarray:
        """
        Scale the samples using the scaler. If no scaler is available yet, fit a new StandardScaler.

        :param samples: Samples to scale.
        :return: Scaled samples.
        """
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(samples)
        else:
            return self.scaler.transform(samples)

    def train(self, n_folds=5, verbose=True):
        """
        Run KFold training. Each fold is a CPTrainer with its own model and scaler.
        """
        # Preprocess samples
        self.preprocess()

        # KFold split on batches (i.e. stages)
        splits = self.kfold_split(n_splits=n_folds, random_state=self.config.get('random_state', 42))

        # Create trainers for each fold
        self.trainers = list()
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_batches = [self._batches[i] for i in train_idx]
            val_batches = [self._batches[i] for i in val_idx]

            # Prepare fold data
            x_train, y_train, train_group_sizes = self.prepare_fold(train_batches)
            x_val, y_val, val_group_sizes = self.prepare_fold(val_batches)

            # Create and train a new CPTrainer for this fold
            fold_trainer = CPTrainer(
                collector=self.collector,
                rider_feature_filter=self.rider_feature_filter,
                stage_feature_filter=self.stage_feature_filter,
                entry_feature_filter=self.entry_feature_filter,
                interactions=self.interactions,
                rider_filter=self.rider_filter,
                stage_filter=self.stage_filter,
                scaler=self.scaler,
                model=type(self.model)(config=self.model.config),
                config=self.config.copy() if self.config else None
            )

            # Manually set processed data for this fold
            fold_trainer._samples = x_train
            fold_trainer._targets = y_train
            fold_trainer._stages = [batch.stages[0] for batch in train_batches]
            fold_trainer._riders = [r for batch in train_batches for r in batch.riders]
            fold_trainer._batches = train_batches

            # Train the model
            fold_trainer.model.train(x_train, train_group_sizes, y_train, verbose=verbose)

            # TODO: Does this make sense? Yes - validates against the remaining fold?
            # Test the model
            fold_trainer.model.test(x_val, val_group_sizes, y_val, verbose=verbose)

            # Plot model
            fold_trainer.plot()

            # Append trainer
            self.trainers.append(fold_trainer)


if __name__ == "__main__":

    from cycling_predictor.models import XGBModel
    from cycling_predictor.collectors import CPGTEntryCollector
    from cycling_predictor.processors import CPPredictor, CPEnsemblePredictor
    import operator as op

    # Load collectors
    _entry_collector = CPGTEntryCollector.load(
        '../collectors/data/CPGTEntryCollector_gts_2023_2024_2025_50.json'
    )
    _prediction_entry_collector = CPGTEntryCollector.load(
        '../collectors/data/CPGTEntryCollector_giro_2026.json'
    )

    # Setup profile and filters
    profile = 'RR3'
    match profile:

        case 'RR3':

            # Setup filters and interactions
            _stage_filter = {'stage_profile': (3,), 'stage_type': ('RR',)}
            _rider_feature_filter = ('cob', 'avg', 'flt', 'mtn', 'gc_', 'pr_', 'tts', 'ttl', 'itt')
            _entry_feature_filter = ('rider_form_mtn', 'is_giro', 'is_tour', 'is_vuelta')
            _interactions = {
                ('spr', 'gradient_final_km'): op.sub,
                ('hll', 'profile_score'): op.add,
                ('hll', 'vertical_meters'): op.add,
            }

            # Model config
            _xgb_model = XGBModel(
                config={
                    'k': 10,
                    'learning_rate': 0.01,
                    'max_depth': 6,
                    # 'reg_alpha': 2,
                    # 'reg_lambda': 3,
                    'reg_alpha': 4,
                    'reg_lambda': 4,
                    'n_estimators': 750,
                }
            )

        case 'RR4':

            # Setup filters and interactions
            _stage_filter = {'stage_profile': (4,), 'stage_type': ('RR',)}
            _rider_feature_filter = ('cob', 'avg', 'flt', 'or_', 'spr', 'pr_', 'tts', 'ttl')
            _stage_feature_filter = ()
            _entry_feature_filter = ('rider_form_flt', 'rider_form_hll', 'is_giro', 'is_tour', 'is_vuelta')
            _interactions = {
                ('mtn', 'profile_score'): op.add,
                ('mtn', 'vertical_meters'): op.add,
            }

            # Model config
            _xgb_model = XGBModel(
                config={
                    'k': 20,
                    'learning_rate': 0.01,
                    'max_depth': 8,
                    # 'reg_alpha': 1,
                    # 'reg_lambda': 2,
                    'reg_alpha': 4,
                    'reg_lambda': 4,
                    'n_estimators': 1500,
                }
            )

        case 'RR5':

            # Setup filters and interactions
            _stage_filter={'stage_profile': (5,), 'stage_type': ('RR',)}
            _rider_feature_filter=('cob', 'avg', 'flt', 'or_', 'spr', 'pr_', 'tts', 'ttl', 'itt')
            _stage_feature_filter=()
            _entry_feature_filter=('rider_form_flt', 'rider_form_hll', 'is_giro', 'is_tour', 'is_vuelta')
            _interactions={
                ('mtn', 'profile_score'): op.add,
                ('mtn', 'vertical_meters'): op.add,
            }

            # Model config
            _xgb_model = XGBModel(
                config={
                    'k': 20,
                    'learning_rate': 0.02,
                    'max_depth': 10,
                    # 'reg_alpha': 4,
                    # 'reg_lambda': 3,
                    'reg_alpha': 4,
                    'reg_lambda': 4,
                    'n_estimators': 1000,
                }
            )

        case _:
            raise ValueError(f"Unknown profile: {profile}")

    # TODO: Implement a KFoldTuner and redo tuning, only allowing higher regularization to prevent overfitting
    # Initialize KFoldTrainer
    _trainer = KFoldTrainer(
        collector=_entry_collector,
        rider_feature_filter=_rider_feature_filter,
        entry_feature_filter=_entry_feature_filter,
        interactions=_interactions,
        stage_filter=_stage_filter,
        model=_xgb_model
    )

    # Run KFold training
    _trainer.train(n_folds=3, verbose=True)

    # Create predictors for the prediction collector
    _predictors = list()
    for _fold_trainer in _trainer.trainers:
        _predictor = CPPredictor(
            collector=_prediction_entry_collector,
            rider_feature_filter=_fold_trainer.rider_feature_filter,
            stage_feature_filter=_fold_trainer.stage_feature_filter,
            entry_feature_filter=_fold_trainer.entry_feature_filter,
            interactions=_fold_trainer.interactions,
            stage_filter=_fold_trainer.stage_filter,
            scaler=_fold_trainer.scaler,
            model=_fold_trainer.model,
        )
        _predictor.preprocess()
        _predictor.predict()
        _predictors.append(_predictor)

    # Ensemble prediction using CPEnsemblePredictor
    _ensemble_predictor = CPEnsemblePredictor(_predictors)
    _ensemble_predictor.predict()
