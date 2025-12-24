# Fairness Checklist for Benchmark Comparisons

## ✅ Verified (All Consistent Across Runs)

1. **Random Seeds**: All experiments use `seed=42`
   - `set_seed(42)` called at start of each `train.main()`
   - Ensures identical RNG state for all random operations

2. **Data Generation**: 
   - `ScalarFieldDataset(10000, seed=42)` - same 10,000 data points
   - Same functional form: `exp(-0.5 * R²) * cos(2 * R²)`
   - Same uniform distribution: [-5, 5] for each coordinate

3. **Train/Val/Test Split**:
   - Uses `torch.Generator().manual_seed(42)` for reproducible split
   - Same 60/20/20 split (6000/2000/2000 samples)
   - Same samples in each split

4. **DataLoader Shuffling**:
   - Uses `torch.Generator().manual_seed(42)` for reproducible shuffling
   - Same batch order within each epoch

5. **Model Initialization**:
   - `set_seed(42)` ensures identical weight initialization
   - Same architecture for same `num_hidden_layers`
   - PyTorch default initialization (Kaiming/He for ReLU)

6. **SO(3) Rotations**:
   - `sample_so3_rotation()` uses `torch.randn()` which respects `torch.manual_seed(42)`
   - Same rotations sampled in same order across runs
   - Verified: Two identical runs produce identical results (difference = 0.00e+00)

7. **Training Hyperparameters**:
   - Learning rate: `3e-4` (consistent)
   - Optimizer: `AdamW` with `weight_decay=0.0` (consistent)
   - Batch size: `128` (consistent)
   - Number of epochs: `100` (consistent)
   - Lambda_sym schedule: 1-cosine from 0.0 to `lambda_sym_max` (varies by experiment, as intended)

8. **Deterministic Operations**:
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`
   - Ensures deterministic CUDA operations (if GPU used)

## ⚠️ Potential Considerations

1. **Device Differences**: 
   - If some runs use CPU and others GPU, there could be minor numerical differences
   - **Status**: All runs use same device (determined by `torch.cuda.is_available()`)
   - **Recommendation**: Ensure consistent device across all benchmark runs

2. **Floating Point Precision**:
   - PyTorch operations are deterministic but floating point arithmetic can have tiny differences
   - **Status**: With deterministic mode enabled, differences should be negligible
   - **Verified**: Test shows identical results (0.00e+00 difference)

3. **Optimizer State**:
   - AdamW maintains internal state (momentum, variance estimates)
   - **Status**: Since model is recreated each run, optimizer state starts fresh
   - **Verified**: Each experiment creates new model and optimizer

## ✅ Conclusion

**All experiments are fair and comparable.** The only intentional differences are:
- `lambda_sym_max` (hyperparameter being tested)
- `symmetry_layer` (hyperparameter being tested)

Everything else is held constant, ensuring fair comparison of these hyperparameters.

