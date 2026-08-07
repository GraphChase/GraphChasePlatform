#!/bin/bash
# Example script showing how to launch each algorithm

# ===== 1. CFRMIX =====
python3 -u -m graphchase.scripts.run_cfrmix \
        --config graphchase/solver_cfgs/cfrmix_cfgs_5_5_1.yaml \
        --set seed=3407

# ===== 2. Grasper =====
python3 -u -m graphchase.scripts.run_grasper_mappo \
        --config graphchase/solver_cfgs/grasper_mappo_cfgs_grid_7_7_1.yaml \
        --mode psro \
        --set seed=3407

# ===== 3. NSGNFSP =====
python3 -u -m graphchase.scripts.run_nsgnfsp \
        --config graphchase/solver_cfgs/nsgnfsp_cfgs_7_7_5_T4.yaml \
        --set seed=3407

# ===== 4. NSGZero =====
# Neural Stochastic Game Zero (similar to AlphaZero)
python3 -u -m graphchase.scripts.run_nsgzero \
        --config graphchase/solver_cfgs/nsgzero_cfgs_7_7_5_T4.yaml

# ===== 5. PretrainPSRO =====
python3 -u -m graphchase.scripts.run_pretrain_psro \
        --config graphchase/solver_cfgs/pretrain_psro_cfgs_timesquare.yaml \
        --set seed=3407

# ===== Tips =====
# - Modify the --config parameter to use different graph configurations
# - Add --set key=value to override config parameters
