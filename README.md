# Latent Symmetry (latsym)

Tools for studying symmetry emergence in neural network latent representations.

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

## Usage

Run the invariance analysis:

```bash
python scripts/analyze_invariance.py
```

Or use the library directly:

```python
from latsym.tasks import create_dataloaders, gaussian_ring
from latsym.models import MLP
from latsym.metrics import get_metric, list_metrics

# List available metrics
print(list_metrics())  # ['Q', ...]

# Get a metric instance
metric = get_metric("Q", n_rotations=32)

# Compute and plot
values = metric.compute(model, data, device=device)
metric.plot(values, save_path="Q_vs_layer.png")
```

## Package Structure

```
src/
├── groups/           # Symmetry group implementations
│   ├── so2.py        # SO(2) rotations
│   ├── so3.py        # SO(3) rotations (stub)
│   └── lorentz.py    # Lorentz group (stub)
├── metrics/          # Invariance metrics
│   ├── base.py       # Metric protocol
│   ├── registry.py   # Auto-discovery
│   └── q_metric.py   # Q orbit variance metric
├── tasks/            # Experiment tasks
│   └── so2_regression.py  # SO(2) scalar field regression
├── models.py         # MLP model
├── train.py          # Training utilities
└── eval.py           # Visualization utilities
```

## Configuration

Configuration files are in `config/` using Hydra. See `config/config.yaml` for defaults.
