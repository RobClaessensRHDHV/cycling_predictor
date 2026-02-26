import operator as op

import numpy as np

from cycling_predictor.collectors import CPGTEntryCollector
from cycling_predictor.models import XGBModel
from cycling_predictor.processors import CPTrainer, CPPredictor


# Get entry collector
_entry_collector = CPGTEntryCollector.load(
    '..\\cycling_predictor\\collectors\\data\\CPClassicEntryCollector_classics_2023_2024_2025_50.json')
# _entry_collector = CPGTEntryCollector.load(
#     r'..\cycling_predictor\collectors\data\CPGTEntryCollector_paris_nice_tirreno_adriatico_2023_2024_2025_50.json')

# Set up trainer
trainer = CPTrainer(
    collector=_entry_collector,
    rider_feature_filter=('pr_', 'tts', 'ttl', 'mtn', 'gc_', 'avg', 'cob'),   # Sprint
    # rider_feature_filter=('pr_', 'tts', 'ttl', 'mtn', 'gc_', 'avg'),          # Cobbles, hills
    # rider_feature_filter=('pr_', 'tts', 'ttl', 'avg', 'cob', 'rider_form'),     # Mountains
    interactions={
        ('spr', 'gradient_final_km'): op.truediv,
        ('spr', 'race_startlist_quality_score'): op.mul,
        # ('cob', 'profile_score'): op.mul,
        # ('cob', 'vertical_meters'): op.mul,
        # ('cob', 'race_startlist_quality_score'): op.mul,
        ('hll', 'gradient_final_km'): op.mul,
        ('hll', 'profile_score'): op.mul,
        ('hll', 'vertical_meters'): op.mul,
        ('hll', 'race_startlist_quality_score'): op.mul,
        # Only for mountains (RR5)
        # ('mtn', 'gradient_final_km'): op.mul,
        # ('mtn', 'profile_score'): op.mul,
        # ('mtn', 'vertical_meters'): op.mul,
        # ('mtn', 'race_startlist_quality_score'): op.mul,
    },
    stage_filter={'stage_type': ('RR',), 'terrain_types': ('sprint',)},
    # stage_filter={'stage_type': ('RR',), 'stage_profile': (5,)},
)

# Preprocess data
trainer.preprocess()

# Initialize model
# Sprint
xgb_model = XGBModel(
    config={
        'k': 10,
        'learning_rate': 0.025,
        'max_depth': 5,
        'reg_alpha': 2,
        'reg_lambda': 3,
        'n_estimators': 500,
    }
)

# # Cobbles
# xgb_model = XGBModel(
#     config={
#         'k': 10,
#         'learning_rate': 0.01,
#         'max_depth': 5,
#         'reg_alpha': 4,
#         'reg_lambda': 2,
#         'n_estimators': 500,
#     }
# )

# Hills / Mountains
xgb_model = XGBModel(
    config={
        'k': 10,
        'learning_rate': 0.03,
        # Lower learning rate for mountains, compensate for low number of samples
        # 'learning_rate': 0.01,
        'max_depth': 5,
        'reg_alpha': 4,
        'reg_lambda': 4,
        'n_estimators': 500,
    }
)

# Set model
trainer.model = xgb_model

# Train model
random_state = 20       # Best CL sprint - no cob (.563 ± .066, .698 ± .038)
# random_state = 15       # Best CL cobbles (.547 ± .103, .692 ± .062)
# random_state = 29       # Best CL hills (.493 ± .186, .657 ± .013)

# random_state = 32       # Best PN/TA sprint - no cobble interactions (.573 ± .003, .576 ± .114)
# random_state = 11       # Best PN/TA mountains - no cob & form - with gc & mtn - mtn interactions (.742 ± .064, .768 ± .097)

# random_state = range(1, 101)
if isinstance(random_state, int):
    trainer.config = {'random_state': random_state}
    trainer.train()
    trainer.plot()
    trainer.dump()
else:
    sr20ps, sr20p_stds, sr20rs, sr20r_stds, sr20s = list(), list(), list(), list(), list()
    for i in random_state:
        trainer.config = {'random_state': i}
        trainer.train()

        if trainer.model.eval_metrics.get('sr20', 0) > max(sr20s, default=0):
            print(f"New best model found with random_state={i}: "
                  f"SR20 pred. {trainer.model.eval_metrics.get('sr20p', 0):.3f} "
                  f"± {trainer.model.eval_metrics.get('sr20p_std', 0):.3f}, "
                  f"SR20 res. {trainer.model.eval_metrics.get('sr20r', 0):.3f} "
                  f"± {trainer.model.eval_metrics.get('sr20r_std', 0):.3f}, "
                  f"SR20 {trainer.model.eval_metrics.get('sr20', 0):.3f}")
            trainer.plot()

        sr20ps.append(trainer.model.eval_metrics.get('sr20p', 0))
        sr20p_stds.append(trainer.model.eval_metrics.get('sr20p_std', 0))
        sr20rs.append(trainer.model.eval_metrics.get('sr20r', 0))
        sr20r_stds.append(trainer.model.eval_metrics.get('sr20r_std', 0))
        sr20s.append(trainer.model.eval_metrics.get('sr20', 0))

# Get prediction entry collector
_prediction_entry_collector = CPGTEntryCollector.load(
    '..\\cycling_predictor\\collectors\\data\\CPClassicEntryCollector_classics_2022_50.json')
# _prediction_entry_collector = CPGTEntryCollector.load(
#     '..\\cycling_predictor\\collectors\\data\\CPGTEntryCollector_paris_nice_tirreno_adriatico_2022_50.json')

# Set up predictor with trained model
predictor = CPPredictor(
    collector=_prediction_entry_collector,
    rider_feature_filter=trainer.rider_feature_filter,
    interactions=trainer.interactions,
    stage_filter=trainer.stage_filter,
    scaler=trainer.scaler,
    model=trainer.model,
)

# Preprocess data for prediction
predictor.preprocess()

# Predict
predictor.predict()
