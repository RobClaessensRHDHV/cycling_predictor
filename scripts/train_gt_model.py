import operator as op

import numpy as np

from cycling_predictor.collectors import CPGTEntryCollector
from cycling_predictor.models import XGBModel
from cycling_predictor.processors import CPTrainer, CPPredictor


# Get entry collector
_entry_collector = CPGTEntryCollector.load(
    '../cycling_predictor/collectors/data/CPGTEntryCollector_gts_2023_2024_2025_100.json'
)

# Setup profile
profile = 'RR1'

# Setup trainer
match profile:
    case 'RR1':
        trainer = CPTrainer(
            collector=_entry_collector,
            rider_feature_filter=('cob', 'mtn', 'gc_'),
            stage_feature_filter=(),
            interactions={
                ('spr', 'gradient_final_km'): op.sub,
            },
            stage_filter={'stage_profile': (1,), 'stage_type': ('RR',)},
        )

    case 'RR2_RR3':
        trainer = CPTrainer(
            collector=_entry_collector,
            rider_feature_filter=('cob', 'mtn', 'gc_'),
            stage_feature_filter=(),
            interactions={
                ('spr', 'gradient_final_km'): op.sub,
                ('hll', 'profile_score'): op.add,
                ('hll', 'vertical_meters'): op.add,
            },
            stage_filter={'stage_profile': (2, 3,), 'stage_type': ('RR',)},
        )

    case 'RR4_RR5':
        trainer = CPTrainer(
            collector=_entry_collector,
            rider_feature_filter=('cob',),
            stage_feature_filter=(),
            interactions={
                # TODO: Check interaction after normalization
                ('spr', 'gradient_final_km'): op.sub,
                ('mtn', 'profile_score'): op.add,
                ('mtn', 'vertical_meters'): op.add,
            },
            stage_filter={'stage_profile': (4, 5,), 'stage_type': ('RR',)},
        )

    case _:
        raise ValueError(f"Unknown profile: {profile}")

# Preprocess data
trainer.preprocess()

# Initialize model
match profile:
    case 'RR1':
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

    case 'RR2_RR3':
        xgb_model = XGBModel(
            config={
                'k': 10,
                'learning_rate': 0.01,
                'max_depth': 5,
                'reg_alpha': 0,
                'reg_lambda': 0,
                'n_estimators': 500,
            }
        )

    case 'RR4_RR5':
        # RR4/5
        xgb_model = XGBModel(
            config={
                'k': 10,
                'learning_rate': 0.01,
                'max_depth': 10,                # Check 8 - 10
                'reg_alpha': 0,
                'reg_lambda': 1,
                'n_estimators': 750,
            }
        )

# Set model
trainer.model = xgb_model

# Train model
# random_state = x   # Best RR1
random_state = range(1, 51)
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

# Get prediction entry collector
_prediction_entry_collector = CPGTEntryCollector.load(
    '../cycling_predictor/collectors/data/CPGTEntryCollector_giro_2026_100.json'
)

# Set up stage filter
match profile:
    case 'RR1':
        stage_filter = {'stage_profile': (1,), 'stage_type': ('RR',)}
    case 'RR2_RR3':
        stage_filter = {'stage_profile': (2, 3,), 'stage_type': ('RR',)}
    case 'RR4_RR5':
        stage_filter = {'stage_profile': (4, 5,), 'stage_type': ('RR',)}
    case _:
        raise ValueError(f"Unknown profile: {profile}")

# Set up predictor with trained model
predictor = CPPredictor(
    collector=_prediction_entry_collector,
    rider_feature_filter=trainer.rider_feature_filter,
    stage_feature_filter=trainer.stage_feature_filter,
    interactions=trainer.interactions,
    stage_filter=stage_filter,
    scaler=trainer.scaler,
    model=trainer.model,
)

# Preprocess data for prediction
predictor.preprocess()

# Predict
predictor.predict()
