# Latent Symmetry

**Encouraging rotational invariance in neural network latent representations via SO(3) regularization.**

## Overview

This project explores whether we can improve neural network generalization by explicitly encouraging symmetry in hidden layer representations. The core idea: if the target function is rotationally invariant, the network's internal representations should also be invariant under rotations.

### The Task

We train an MLP to learn a rotationally invariant 3D scalar field:

```
f(x, y, z) = exp(-0.5 * R²) * cos(2 * R²)
```

where `R² = x² + y² + z²`. Since the function only depends on the distance from the origin, it's invariant under any SO(3) rotation.

### The Method

We add a **latent symmetry loss** that penalizes variance in intermediate representations under random rotations:

1. Sample two random SO(3) rotations R₁, R₂
2. Rotate input batch: x₁ = R₁x, x₂ = R₂x  
3. Get intermediate activations: h₁ = φᵢ(x₁), h₂ = φᵢ(x₂)
4. Minimize: `L_sym = ½ * E[‖h₁ - h₂‖²]`

This encourages the network to learn representations that are invariant to input rotations, without requiring data augmentation.

## Project Structure

```
├── models.py          # MLP with intermediate activation extraction
├── symmetry.py        # SO(3) sampling and orbit variance loss
├── data.py            # Scalar field dataset generation
├── train.py           # Training loop with symmetry regularization
├── benchmark.py       # Hyperparameter grid search
├── analyze_benchmark.py   # Benchmark result analysis
├── compare_lambdas.py # Visualization of results across λ values
├── plots.py           # Training visualization utilities
├── tests.py           # Unit tests
└── results/           # Benchmark CSV outputs
```

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch, NumPy, Matplotlib, tqdm

## Usage

### Training a single model

```bash
# Basic training (with symmetry regularization)
python train.py

# Customize hyperparameters
python train.py --symmetry-layer -1 --lambda-sym-max 1.0 --num-hidden-layers 6 --hidden-dim 128

# Disable symmetry regularization (baseline)
python train.py --symmetry-layer None --lambda-sym-max 0.0

# Run headless (no plots)
python train.py --headless
```

**Key arguments:**
- `--symmetry-layer`: Layer index for symmetry loss (-1 = output, 1-N = hidden layers, None = disabled)
- `--lambda-sym-max`: Maximum λ_sym value (cosine schedule from 0 to this value)
- `--num-hidden-layers`: Number of hidden layers
- `--hidden-dim`: Width of hidden layers
- `--learning-rate`: Learning rate (default: 3e-4)
- `--run-seed`: Random seed for reproducibility

### Running benchmarks

```bash
# Run full benchmark with default settings
python benchmark.py

# Customize architecture
python benchmark.py --num-hidden-layers 4 --hidden-dim 256

# Run multiple seeds
python benchmark.py --seeds 42 43 44
python benchmark.py --seeds 1-100    # Range notation
```

Results are saved incrementally to `results/` as CSV files.

### Analyzing results

```bash
python analyze_benchmark.py
```

## Key Concepts

### Symmetry Layer Selection

The `symmetry_layer` parameter controls where the invariance constraint is applied:

- **Layer -1 (output)**: Encourages the final prediction to be rotation-invariant
- **Layer 1-N (hidden)**: Encourages intermediate representations to be invariant
- **None**: No symmetry regularization (baseline)

### λ_sym Schedule

The symmetry loss weight follows a 1-cosine warmup schedule:
```
λ(t) = λ_max * (1 - cos(π * t / T)) / 2
```
This starts at 0 and smoothly increases to `lambda_sym_max`, allowing the network to first learn useful features before enforcing invariance.

### Reproducibility

The training pipeline uses deterministic seeding with three derived seeds:
- **Data seed**: Controls dataset generation and splits
- **Model seed**: Controls weight initialization
- **Augmentation seed**: Controls SO(3) rotation sampling

See `FAIRNESS_CHECK.md` for verification that experiments are fairly comparable.

## Running Tests

```bash
pytest tests.py -v
```

Tests cover:
- MLP forward pass and intermediate activation extraction
- SO(3) rotation sampling (validates Haar uniformity)
- Seed reproducibility across data, model, and augmentation

## Example Results

After training with symmetry regularization on a 6-layer MLP:

| Configuration | Test Task Loss | Test Sym Loss |
|--------------|----------------|---------------|
| Baseline (λ=0) | ~5e-4 | N/A |
| λ_max=1.0, layer=-1 | ~3e-4 | ~1e-5 |

The symmetry loss encourages learning of genuinely invariant representations, often improving generalization.

## License

Research code — feel free to use and adapt.

