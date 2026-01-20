# SO2 Toy

Binary classification toy task for studying symmetry emergence in neural networks.

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

After installation, you can use the command-line scripts:

```bash
# Train a model
so2-toy-train

# Plot the dataset
so2-toy-plot-data
```

Or use the scripts directly:

```bash
# Train
python scripts/train.py

# Plot data
python scripts/plot_data.py
```

## Package Structure

The package source files are in `src/`:
- `src/data.py`: Dataset generation (TwoCloudDataset)
- `src/models.py`: MLP model implementation
- `src/train.py`: Training utilities
- `src/eval.py`: Evaluation utilities (placeholder)
- `src/scripts/`: Command-line scripts

When installed, import as:
```python
from so2toy.data import TwoCloudDataset
from so2toy.models import MLP
from so2toy.train import train_loop
```

## Configuration

Configuration files are in `config/` using Hydra. See `config/config.yaml` for defaults.
