import operator as op

import numpy as np

from cycling_predictor.collectors import CPGTEntryCollector
from cycling_predictor.processors.trainer import CPTrainer
from cycling_predictor.processors.predictor import CPPredictor
from cycling_predictor.models import XGBModel


# Get entry collector
_entry_collector = CPGTEntryCollector.load(
    '../collectors/data/entry_collector_giro_tour_vuelta_2023_2024_2025_100.json')

# Set up trainer (RR1)
trainer = CPTrainer(
    collector=_entry_collector,
    rider_feature_filter=('pr_', 'tts', 'ttl', 'cob', 'mtn', 'gc_'),
    stage_feature_filter=('race_startlist_quality_score',),
    interactions={
        ('spr', 'gradient_final_km'): op.sub,
    },
    stage_filter={'stage_profile': (1,), 'stage_type': ('RR',)},
)

# # Set up trainer (RR2 & RR3)
# trainer = CPTrainer(
#     collector=_entry_collector,
#     rider_feature_filter=('pr_', 'tts', 'ttl', 'cob', 'mtn', 'gc_'),
#     stage_feature_filter=('race_startlist_quality_score',),
#     interactions={
#         ('spr', 'gradient_final_km'): op.sub,
#         ('hll', 'profile_score'): op.add,
#         ('hll', 'vertical_meters'): op.add,
#     },
#     stage_filter={'stage_profile': (2, 3,), 'stage_type': ('RR',)},
# )

# # Set up trainer (RR4 & RR5)
# trainer = CPTrainer(
#     collector=_entry_collector,
#     rider_feature_filter=('pr_', 'tts', 'ttl', 'flt', 'cob'),
#     stage_feature_filter=('race_startlist_quality_score',),
#     interactions={
#         # TODO: Check interaction after normalization
#         ('spr', 'gradient_final_km'): op.sub,
#         ('hll', 'profile_score'): op.add,
#         ('hll', 'vertical_meters'): op.add,
#         ('mtn', 'profile_score'): op.add,
#         ('mtn', 'vertical_meters'): op.add,
#     },
#     stage_filter={'stage_profile': (4, 5,), 'stage_type': ('RR',)},
# )

# Preprocess data
trainer.preprocess()

# Initialize model
# RR1
xgb_model = XGBModel(
    config={
        'k': 10,
        'learning_rate': 0.01,
        'max_depth': 8,                 # 7 - 8
        'reg_alpha': 1,
        'reg_lambda': 1,
        'n_estimators': 1000,            # 500 - 1000
    }
)

# # RR2/3
# xgb_model = XGBModel(
#     config={
#         'k': 10,
#         'learning_rate': 0.01,
#         'max_depth': 5,
#         'reg_alpha': 0,
#         'reg_lambda': 0,
#         'n_estimators': 500,
#     }
# )

# # RR4/5
# xgb_model = XGBModel(
#     config={
#         'k': 10,
#         'learning_rate': 0.01,
#         'max_depth': 10,                # Check 8 - 10
#         'reg_alpha': 0,
#         'reg_lambda': 1,
#         'n_estimators': 750,
#     }
# )

# Set model
trainer.model = xgb_model

# Train model
random_state = 16   # Best RR1
# random_state = 22   # Best RR1/2
# random_state = 7    # Best RR2/3
# random_state = 9    # Best RR4/5
# random_state = range(1, 51)
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
            trainer.plot()

        sr20ps.append(trainer.model.eval_metrics.get('sr20p', 0))
        sr20p_stds.append(trainer.model.eval_metrics.get('sr20p_std', 0))
        sr20rs.append(trainer.model.eval_metrics.get('sr20r', 0))
        sr20r_stds.append(trainer.model.eval_metrics.get('sr20r_std', 0))
        sr20s.append(trainer.model.eval_metrics.get('sr20', 0))

    print('-----')
    print(f"Avg SR20 pred.: {np.mean(sr20ps):.3f} ± {np.std(sr20ps):.3f}")
    print(f"Max SR20 pred.: {max(sr20ps):.3f}")
    print(f"Avg SR20 pred. Std: {np.mean(sr20p_stds):.3f} ± {np.std(sr20p_stds):.3f}")
    print(f"Min SR20 pred. Std: {min(sr20p_stds):.3f}")
    print('-----')
    print(f"Avg SR20 res.: {np.mean(sr20rs):.3f} ± {np.std(sr20rs):.3f}")
    print(f"Max SR20 res.: {max(sr20rs):.3f}")
    print(f"Avg SR20 res. Std: {np.mean(sr20r_stds):.3f} ± {np.std(sr20r_stds):.3f}")
    print(f"Min SR20 res. Std: {min(sr20r_stds):.3f}")
    print('-----')
    print(f"Avg SR20: {np.mean(sr20s):.3f} ± {np.std(sr20s):.3f}")
    print(f"Max SR20: {max(sr20s):.3f}")

# Set up predictor with trained model
predictor = CPPredictor(
    collector=_entry_collector,
    rider_feature_filter=trainer.rider_feature_filter,
    stage_feature_filter=trainer.stage_feature_filter,
    interactions=trainer.interactions,
    stage_filter={'name': ('tour-de-france',), 'year': (2025,), 'stage_profile': (1,), 'stage_type': ('RR',)},
    # stage_filter={'name': ('tour-de-france',), 'year': (2025,), 'stage_profile': (2, 3,), 'stage_type': ('RR',)},
    # stage_filter={'name': ('tour-de-france',), 'year': (2025,), 'stage_profile': (4, 5,), 'stage_type': ('RR',)},
    scaler=trainer.scaler,
    model=trainer.model,
)

# Preprocess data for prediction
predictor.preprocess()

# Predict
predictor.predict()
