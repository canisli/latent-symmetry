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
python scripts/benchmark_penalties.py --penalty-types Q_h_ns --seeds 1-50
python scripts/benchmark_penalties.py --penalty-types Q_z_ns --seeds 1-50
python scripts/benchmark_penalties.py --penalty-types Q_z --seeds 1-50
python scripts/benchmark_penalties.py --penalty-types N_h --layers -1 --seeds 1-100
python scripts/plot_benchmark.py results/Q_h_penalty/ --layers n123l
python scripts/plot_benchmark.py results/N_h_penalty/ --layers bl
python scripts/benchmark_penalties.py --penalty-types N_h  --seeds 1-100 --lambda-values 0.0 0.001 0.01 0.1 1.0 10.0 100.0 1000.0 10000.0