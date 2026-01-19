from typing import Union, List
import operator as op

import numpy as np
import optuna

from cycling_predictor.collectors import CPGTEntryCollector
from cycling_predictor.models import XGBModel
from cycling_predictor.processors import CPTrainer


def param_space(trial: optuna.Trial) -> dict:
    k = 10
    lr = trial.suggest_float('learning_rate', 0.01, 0.03, step=0.005)
    md = 5
    al = trial.suggest_int('reg_alpha', 1, 6)
    la = trial.suggest_int('reg_lambda', 1, 6)
    # ne = trial.suggest_categorical('n_estimators', [500, 750, 1000])
    ne = 500

    return {
        'k': k,
        'learning_rate': lr,
        'max_depth': md,
        'reg_alpha': al,
        'reg_lambda': la,
        'n_estimators': ne,
    }


def objective(trial: optuna.Trial) -> float:

    # Set random state
    random_state: Union[int, List[int]] = list(range(0, 51, 5))

    # Initialize model
    model = XGBModel(
        config=param_space(trial)
    )

    # Set model
    trainer.model = model

    # Train model
    sr20ps, sr20p_stds, sr20rs, sr20r_stds, sr20s, sr20_spreads = list(), list(), list(), list(), list(), list()
    for rs in random_state:
        trainer.config = {'random_state': rs}
        trainer.train(verbose=False)

        sr20ps.append(trainer.model.eval_metrics.get('sr20p', 0))
        sr20p_stds.append(trainer.model.eval_metrics.get('sr20p_std', 0))
        sr20rs.append(trainer.model.eval_metrics.get('sr20r', 0))
        sr20r_stds.append(trainer.model.eval_metrics.get('sr20r_std', 0))
        sr20s.append(trainer.model.eval_metrics.get('sr20', 0))

        if trainer.model.eval_metrics.get('sr20p', 0) > trainer.model.eval_metrics.get('sr20r', 0):
            sr20_spreads.append(trainer.model.eval_metrics.get('sr20p') - trainer.model.eval_metrics.get('sr20r'))
        elif trainer.model.eval_metrics.get('sr20p', 0) < trainer.model.eval_metrics.get('sr20r', 0):
            sr20_spreads.append(trainer.model.eval_metrics.get('sr20r') - trainer.model.eval_metrics.get('sr20p'))
        else:
            print("SR20p or SR20r is zero, cannot compute spread, spread skipped.")

    print(f"\nTrial {trial.number}: Avg SR20: {np.mean(sr20s):.3f} ± {np.std(sr20s):.3f}"
          f"\n\t(Avg spread: {np.mean(sr20_spreads):.3f} ± {np.std(sr20_spreads):.3f})"
          f"\n\t(Pred. Std SR20: {np.mean(sr20p_stds):.3f} ± {np.std(sr20p_stds):.3f})"
          f"\n\t(Res. Std SR20: {np.mean(sr20r_stds):.3f} ± {np.std(sr20r_stds):.3f})")

    trial.set_user_attr('sr20p_std_mean', float(np.mean(sr20p_stds)))
    trial.set_user_attr('sr20r_std_mean', float(np.mean(sr20r_stds)))
    trial.set_user_attr('sr20_spr_mean', float(np.mean(sr20_spreads)))
    return float(np.mean(sr20s) - np.mean(sr20_spreads))


# Get entry collector
_entry_collector = CPGTEntryCollector.load(
    '..\\cycling_predictor\\collectors\\data\\entry_collector_classics_2023_2024_2025_50.json')

# Set up trainer
trainer = CPTrainer(
    collector=_entry_collector,
    rider_feature_filter=('pr_', 'tts', 'ttl', 'mtn', 'gc_', 'avg'),
    interactions={
        ('spr', 'gradient_final_km'): op.truediv,
        ('spr', 'race_startlist_quality_score'): op.mul,
        ('cob', 'profile_score'): op.mul,
        ('cob', 'vertical_meters'): op.mul,
        ('cob', 'race_startlist_quality_score'): op.mul,
        ('hll', 'gradient_final_km'): op.mul,
        ('hll', 'profile_score'): op.mul,
        ('hll', 'vertical_meters'): op.mul,
        ('hll', 'race_startlist_quality_score'): op.mul,
    },
    # stage_filter={'race_profile': ('cobbles',)},
)

# Preprocess data
trainer.preprocess()

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print("Best trial:")
print(f"Spearman's Rho (20): {study.best_trial.value} ({study.best_trial.user_attrs['sr20_spr_mean']}) ± {study.best_trial.user_attrs['sr20p_std_mean']} ± {study.best_trial.user_attrs['sr20r_std_mean']}")
print(f"Hyperparameters: {study.best_trial.params}")
