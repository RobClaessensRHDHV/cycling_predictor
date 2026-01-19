import operator as op

import numpy as np

from cycling_predictor.collectors import CPGTEntryCollector
from cycling_predictor.processors.trainer import CPTrainer
from cycling_predictor.processors.predictor import CPPredictor
from cycling_predictor.models import XGBModel


# Get entry collector
_entry_collector = CPGTEntryCollector.load(
    '..\collectors\data\entry_collector_classics_2023_2024_2025_50.json')

# Set up trainer
trainer = CPTrainer(
    collector=_entry_collector,
    rider_feature_filter=('pr_', 'tts', 'ttl', 'mtn', 'gc_', 'avg'),
    # rider_feature_filter=('pr_', 'tts', 'ttl', 'mtn', 'gc_', 'flt', 'avg'),
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
    stage_filter={'race_profile': ('hills',)},
)

# Preprocess data
trainer.preprocess()

# Initialize model
# # Sprint
# xgb_model = XGBModel(
#     config={
#         'k': 10,
#         'learning_rate': 0.025,
#         'max_depth': 5,
#         'reg_alpha': 2,
#         'reg_lambda': 3,
#         'n_estimators': 500,
#     }
# )

# # Cobbles (no avg)
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

# Hills (no avg)
xgb_model = XGBModel(
    config={
        'k': 10,
        'learning_rate': 0.03,
        'max_depth': 5,
        'reg_alpha': 4,
        'reg_lambda': 4,
        'n_estimators': 500,
    }
)

# # Overall - no avg
# xgb_model = XGBModel(
#     config={
#         'k': 10,
#         'learning_rate': 0.02,
#         'max_depth': 5,
#         'reg_alpha': 0,
#         'reg_lambda': 1,
#         'n_estimators': 500,
#     }
# )

# Set model
trainer.model = xgb_model

# Train model
# random_state = 20       # Best CL sprint - interactions - no avg / flt (.473 ± .104, .771 ± .029)
# random_state = 20       # Best CL sprint - interactions - no avg (.563 ± .032, .702 ± .036)
# random_state = 15       # Best CL cobbles - interactions - no avg (.678 ± .152, .691 ± .062)
random_state = 29       # Best CL hills - interactions - no avg (.524 ± .195, .651 ± .032)
# random_state = 18       # Best CL overall - interactions - no avg (.xxx ± 0.xxx, .xxx ± .xxx)

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
    print('-----')

# Get prediction entry collector
_prediction_entry_collector = CPGTEntryCollector.load(
    '..\collectors\data\entry_collector_classics_2022_50.json')

# Set up predictor with trained model
predictor = CPPredictor(
    collector=_prediction_entry_collector,
    rider_feature_filter=trainer.rider_feature_filter,
    interactions=trainer.interactions,
    # stage_filter={'year': (2022,)},
    stage_filter={'year': (2022,), 'race_profile': ('hills',)},
    scaler=trainer.scaler,
    model=trainer.model,
)

# Preprocess data for prediction
predictor.preprocess()

# Predict
predictor.predict()
