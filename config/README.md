# Hydra Configuration Files

This directory contains Hydra configuration files for training models.

## Structure

- `config.yaml`: Base configuration with common parameters
- `model/transformer.yaml`: Transformer-specific hyperparameters
- `model/deepsets.yaml`: DeepSets-specific hyperparameters

## Usage

### Basic usage (default transformer):
```bash
python train.py
```

### Switch to DeepSets:
```bash
python train.py model=deepsets
```

### Override parameters:
```bash
# Override learning rate
python train.py training.learning_rate=0.0005

# Override model-specific parameters
python train.py model=transformer model.num_blocks=8 model.hidden_channels=512

# Override data parameters
python train.py data.num_events=100000 data.batch_size=128

# Combine multiple overrides
python train.py model=deepsets training.learning_rate=0.001 data.num_events=100000
```

### Run in headless mode:
```bash
python train.py headless=true
```

## Configuration Files

### config.yaml
Contains:
- Data parameters (num_events, n_particles, batch_size, etc.)
- Training parameters (learning_rate, num_epochs, weight_decay, etc.)
- General settings (run_seed, headless, outputs_dir)

### model/transformer.yaml
Contains:
- Model type: transformer
- Architecture: num_blocks, hidden_channels, num_heads
- Transformer-specific: use_mean_pooling, multi_query, etc.

### model/deepsets.yaml
Contains:
- Model type: deepsets
- Architecture: num_blocks, hidden_channels
- DeepSets-specific: pool_mode

## Outputs

Hydra will create output directories in `./outputs/` with timestamps. Each run gets its own directory containing:
- The resolved config
- Logs
- Any saved checkpoints/models

