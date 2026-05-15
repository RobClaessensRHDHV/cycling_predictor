from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import optuna

from cycling_predictor.processors.kfold_processor import KFoldProcessor
from cycling_predictor.models import BaseModel
from cycling_predictor.collectors import CPEntryCollector


class KFoldTuner(KFoldProcessor):
    """
    KFoldTuner for hyperparameter tuning using KFold cross-validation.
    Evaluates parameter sets and returns the best set based on average validation score.
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

    @property
    def dump_fn(self) -> str:
        collector_fn = self.collector.dump_fn.split('_',1)[1]
        return (f"{self.__class__.__name__}_{Path(collector_fn).stem}_"
                f"{self.config.get('test_size')}_{self.config.get('random_state')}_"
                f"{'_'.join([str(v) for val in self.stage_filter.values() for v in val])
                if self.stage_filter else list()}.json")

    @staticmethod
    def parameter_space(trial):
        """
        Defines the Optuna parameter search space.
        """
        k = trial.suggest_categorical('k', [10, 20])
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.03, step=0.01)
        max_depth = trial.suggest_int('max_depth', 5, 10)
        # reg_alpha = trial.suggest_int('reg_alpha', 0, 4)
        # reg_lambda = trial.suggest_int('reg_lambda', 0, 4)
        reg_alpha = trial.suggest_int('reg_alpha', 3, 6)
        reg_lambda = trial.suggest_int('reg_lambda', 3, 6)
        n_estimators = trial.suggest_categorical('n_estimators', [500, 750, 1000])

        return {
            'k': k,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'n_estimators': n_estimators
        }

    @staticmethod
    def create_objective(processor, n_folds=3, random_state=None):
        """
        Returns an Optuna objective function for KFold CV on the processor for a specific random_state.
        Collects sr20, sr20p, sr20r from model and averages them across folds.
        """
        def objective(trial):
            # Suggest hyperparameters
            param_set = KFoldTuner.parameter_space(trial)

            # Create KFold splits for this random state
            splits = processor.kfold_split(n_splits=n_folds, random_state=random_state)
            fold_sr20s = []
            fold_sr20ps = []
            fold_sr20rs = []

            # Train and evaluate for each fold
            for train_idx, val_idx in splits:

                # Prepare data for this fold
                train_batches = [processor._batches[i] for i in train_idx]
                val_batches = [processor._batches[i] for i in val_idx]
                x_train, y_train, train_group_sizes = processor.prepare_fold(train_batches)
                x_val, y_val, val_group_sizes = processor.prepare_fold(val_batches)

                # Create model, train and test
                model = type(processor.model)(config={**processor.model.config, **param_set})
                model.train(x_train, train_group_sizes, y_train, verbose=False)
                model.test(x_val, val_group_sizes, y_val, verbose=False)

                # Collect metrics from model
                fold_sr20s.append(float(model.eval_metrics.get('sr20', 0)))
                fold_sr20ps.append(float(model.eval_metrics.get('sr20p', 0)))
                fold_sr20rs.append(float(model.eval_metrics.get('sr20r', 0)))

            # Average metrics across folds
            avg_sr20 = np.mean(fold_sr20s)
            avg_sr20p = np.mean(fold_sr20ps)
            avg_sr20r = np.mean(fold_sr20rs)

            # Set user attributes for reporting
            trial.set_user_attr('cv_sr20_mean', avg_sr20)
            trial.set_user_attr('cv_sr20p_mean', avg_sr20p)
            trial.set_user_attr('cv_sr20r_mean', avg_sr20r)
            trial.set_user_attr('cv_sr20_std', float(np.std(fold_sr20s)))

            return avg_sr20  # Optuna maximizes

        return objective

    def tune(self, n_folds=3, n_trials=50, verbose=True) -> dict:
        """
        Tune hyperparameters using Optuna with KFold cross-validation for multiple random states.
        For each random state, a separate Optuna study is run. Always uses Spearman's rho as the metric.
        :param n_folds: Number of folds.
        :param n_trials: Number of Optuna trials per random state.
        :param verbose: Print progress.
        :return: Dict with best_params_per_state, best_scores_per_state, optuna_studies
        """
        # Preprocess samples
        self.preprocess()

        # Initiate lists to store results
        best_params_per_state = list()
        best_scores_per_state = list()
        optuna_studies = list()

        # Run Optuna study for each random state
        for inner_random_state in range(1, 6):

            # Create objective and run Optuna study
            objective = KFoldTuner.create_objective(self, n_folds=n_folds, random_state=inner_random_state)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

            # Store results
            best_params = study.best_params
            best_score = study.best_value
            best_trial = study.best_trial
            best_sr20 = best_trial.user_attrs.get('cv_sr20_mean', 0)
            best_sr20p = best_trial.user_attrs.get('cv_sr20p_mean', 0)
            best_sr20r = best_trial.user_attrs.get('cv_sr20r_mean', 0)

            best_params_per_state.append({'random_state': inner_random_state, 'best_params': best_params})
            best_scores_per_state.append({'random_state': inner_random_state, 'best_score': best_score})
            optuna_studies.append(study)

            # Print results for this study
            print(f"\nRandom state: {inner_random_state}")
            print(f"\tBest params: {best_params}")
            print(f"\tBest Spearman's Rho (SR20): {best_sr20:.3f}")
            print(f"\tSR20p (pred top 20): {best_sr20p:.3f}")
            print(f"\tSR20r (result top 20): {best_sr20r:.3f}\n")

        # Find and print the overall best result
        all_best = max(best_scores_per_state, key=lambda d: d['best_score'])
        idx = best_scores_per_state.index(all_best)
        best_trial = optuna_studies[idx].best_trial
        print(f"\nOverall best - Random state: {best_params_per_state[idx]['random_state']}")
        print(f"\tBest params: {best_params_per_state[idx]['best_params']}")
        print(f"\tBest Spearman's Rho (SR20): {best_trial.user_attrs.get('cv_sr20_mean', 0):.3f}")
        print(f"\tSR20p (pred top 20): {best_trial.user_attrs.get('cv_sr20p_mean', 0):.3f}")
        print(f"\tSR20r (result top 20): {best_trial.user_attrs.get('cv_sr20r_mean', 0):.3f}")

        return {
            'best_params_per_state': best_params_per_state,
            'best_scores_per_state': best_scores_per_state,
            'optuna_studies': optuna_studies
        }


if __name__ == "__main__":

    import operator as op
    from cycling_predictor.models import XGBModel
    from cycling_predictor.collectors import CPGTEntryCollector

    # Load collector
    _entry_collector = CPGTEntryCollector.load(
        '../collectors/data/CPGTEntryCollector_gts_2023_2024_2025_50.json'
    )

    # Setup profile and filters
    profile = 'RR1'
    match profile:

        case 'RR1':

            # Setup filters and interactions
            _stage_filter = {'stage_profile': (1,), 'stage_type': ('RR',)}
            _rider_feature_filter = ('cob', 'mtn', 'gc_', 'pr_', 'tts', 'ttl')
            _stage_feature_filter = ()
            _entry_feature_filter = ('rider_form_mtn',)
            _interactions = {
                ('spr', 'gradient_final_km'): op.sub,
                ('hll', 'profile_score'): op.add,
                ('hll', 'vertical_meters'): op.add,
            }

        case 'RR1_RR2':

            # Setup filters and interactions
            _stage_filter = {'stage_profile': (1, 2), 'stage_type': ('RR',)}
            _rider_feature_filter = ('cob', 'mtn', 'gc_', 'pr_', 'tts', 'ttl')
            _stage_feature_filter = ()
            _entry_feature_filter = ('rider_form_mtn',)
            _interactions = {
                ('spr', 'gradient_final_km'): op.sub,
                ('hll', 'profile_score'): op.add,
                ('hll', 'vertical_meters'): op.add,
            }

        case 'RR2':

            # Setup filters and interactions
            _stage_filter = {'stage_profile': (2,), 'stage_type': ('RR',)}
            _rider_feature_filter = ('cob', 'mtn', 'gc_', 'pr_', 'tts', 'ttl')
            _stage_feature_filter = ()
            _entry_feature_filter = ('rider_form_mtn',)
            _interactions = {
                ('spr', 'gradient_final_km'): op.sub,
                ('hll', 'profile_score'): op.add,
                ('hll', 'vertical_meters'): op.add,
            }

        case 'RR2_RR3':

            # Setup filters and interactions
            _stage_filter = {'stage_profile': (3,), 'stage_type': ('RR',)}
            _rider_feature_filter = ('cob', 'avg', 'mtn', 'gc_', 'pr_', 'tts', 'ttl', 'itt')
            _entry_feature_filter = ('rider_form_mtn', 'is_giro', 'is_tour', 'is_vuelta')
            _interactions = {
                ('spr', 'gradient_final_km'): op.sub,
                ('hll', 'profile_score'): op.add,
                ('hll', 'vertical_meters'): op.add,
            }

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

        case 'RR5':

            # Setup filters and interactions
            _stage_filter = {'stage_profile': (5,), 'stage_type': ('RR',)}
            _rider_feature_filter = ('cob', 'avg', 'flt', 'or_', 'spr', 'pr_', 'tts', 'ttl', 'itt')
            _stage_feature_filter = ()
            _entry_feature_filter = ('rider_form_flt', 'rider_form_hll', 'is_giro', 'is_tour', 'is_vuelta')
            _interactions = {
                ('mtn', 'profile_score'): op.add,
                ('mtn', 'vertical_meters'): op.add,
            }

        case 'RR':

            # Setup filters and interactions
            _stage_filter = {'stage_type': ('RR',)}
            _rider_feature_filter = ('cob', 'pr_', 'tts', 'ttl')
            _stage_feature_filter = ()
            _entry_feature_filter = ()
            _interactions = {
                ('spr', 'gradient_final_km'): op.sub,
                ('hll', 'profile_score'): op.add,
                ('hll', 'vertical_meters'): op.add,
                ('mtn', 'profile_score'): op.add,
                ('mtn', 'vertical_meters'): op.add,
            }

        case 'ITT1_ITT2':

            # Setup filters and interactions
            _stage_filter = {'stage_profile': (1, 2), 'stage_type': ('ITT',)}
            _rider_feature_filter = ('cob', 'mtn', 'gc_', 'spr')
            _stage_feature_filter = ()
            _entry_feature_filter = ('rider_form_mtn', 'is_giro', 'is_tour', 'is_vuelta')
            _interactions = {
                ('hll', 'profile_score'): op.add,
                ('hll', 'vertical_meters'): op.add,
            }

        case _:
            raise ValueError(f"Unknown profile: {profile}")

    # Initialize KFoldTuner
    tuner = KFoldTuner(
        collector=_entry_collector,
        rider_feature_filter=_rider_feature_filter,
        entry_feature_filter=_entry_feature_filter,
        interactions=_interactions,
        stage_filter=_stage_filter,
        model=XGBModel()
    )

    # Run tuning
    results = tuner.tune(n_folds=3, verbose=True)
