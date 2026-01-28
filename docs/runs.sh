python scripts/analyze_invariance.py untrained
# python scripts/analyze_invariance.py compare_depths
python scripts/analyze_invariance.py field_survey
python scripts/analyze_invariance.py bessel_radial_mix


# 1. Symmetry penalty on final layer
python scripts/train.py train.lambda_sym=0.01 train.sym_layers=[-1] train.sym_penalty_type=Q_h

# 2. Save frames every 10 seconds
python scripts/train.py train.dynamics_mode=true train.dynamics_interval=10

# 3. 1 + 2
python scripts/train.py train.lambda_sym=0.01 train.sym_layers=[-1] train.sym_penalty_type=Q_h train.dynamics_mode=true train.dynamics_interval=10

# 4. With gradient alignment tracking (for barrier hypothesis testing)
python scripts/train.py train.lambda_sym=0.01 train.sym_layers=[-1] train.sym_penalty_type=Q_h train.grad_align_interval=10

# 5. Make a movie
python scripts/make_gif.py <fdir>

# 6. Benchmark penalties
python scripts/benchmark_penalties.py