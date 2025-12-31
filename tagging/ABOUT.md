# Essential Files for Top Tagging (Python & Dataset Only)

This document lists **only the Python and dataset files** immediately relevant to top tagging, **excluding LLoCa models and configuration files**.

## Core Dataset & Feature Computation Files

### Dataset Loading
- **`experiments/tagging/dataset.py`** ⭐ **ESSENTIAL**
  - `TopTaggingDataset` class for loading `.npz` format top tagging data
  - Handles four-momenta and labels, converts to torch_geometric Data format
  - Independent of LLoCa models

### HEP Utilities
- **`experiments/hep.py`** ⭐ **ESSENTIAL**
  - Pure HEP physics utilities: `get_pt()`, `get_phi()`, `get_eta()`, `get_deltaR()`
  - Four-momentum conversions (E,px,py,pz ↔ pt,phi,eta,mass)
  - No dependencies on LLoCa

### Feature Computation & Embedding
- **`experiments/tagging/embedding.py`** ⚠️ **PARTIALLY USEFUL**
  - **Core functions (independent of LLoCa):**
    - `get_tagging_features()` - Computes 7 standard tagging features (log_pt, log_energy, log_pt_rel, log_energy_rel, dphi, deta, dr)
    - `get_num_tagging_features()` - Returns number of features based on feature type
    - `get_spurion()` - Creates reference spurion vectors (beam, time references)
    - `TAGGING_FEATURES_PREPROCESSING` - Standardization constants (weaver defaults)
    - `dense_to_sparse_jet()` - Converts dense to sparse jet representation
  
  - **Functions with LLoCa dependencies (need adaptation):**
    - `embed_tagging_data()` - Uses `utils.get_batch_from_ptr()`, `lorentz.lorentz_squarednorm()`, `polar_decomposition.restframe_boost()`
    - These can be replaced with simple implementations if you don't need the exact LLoCa behavior

## Files to Skip (Tied to LLoCa Infrastructure)

### Experiment Infrastructure (skip - tied to their setup)
- `experiments/base_experiment.py` - Uses Hydra, MLflow, their training loop
- `experiments/tagging/experiment.py` - Uses BaseExperiment, Hydra configs, LLoCa models
- `experiments/tagging/wrappers.py` - Wraps LLoCa models (TransformerWrapper, etc.)
- `experiments/logger.py`, `experiments/mlflow.py`, `experiments/misc.py` - Infrastructure utilities

### LLoCa Library (skip - you're using your own models)
- Entire `lloca/` directory - Not needed if using your own models

### Configuration Files (skip - you'll use your own configs)
- All files in `config/` and `config_quick/` directories

### Entry Points (skip - you'll write your own)
- `run.py` - Uses Hydra and their experiment setup

## Minimal Essential File List

**For a barebones top tagging testbed with your own models:**

1. **`experiments/tagging/dataset.py`** - Dataset loader
2. **`experiments/hep.py`** - HEP physics utilities  
3. **`experiments/tagging/embedding.py`** - Feature computation (with minor adaptations)

**Total: 3 Python files**

## What You Get From These Files

### From `dataset.py`:
- Loading `.npz` files with structure: `kinematics_{train/test/val}`, `labels_{train/test/val}`
- Converting to torch_geometric `Data` format
- Handling variable-length jets (zero-padding removal)

### From `hep.py`:
- `get_pt(fourmomenta)` - Transverse momentum
- `get_phi(fourmomenta)` - Azimuthal angle  
- `get_eta(fourmomenta)` - Rapidity/pseudorapidity
- `get_deltaR(v1, v2)` - Delta-R distance
- `EPPP_to_PtPhiEtaM2()` / `PtPhiEtaM2_to_EPPP()` - Coordinate conversions

### From `embedding.py`:
- `get_tagging_features(fourmomenta, jet, tagging_features="all")` - Returns 7 standardized features:
  1. log_pt (standardized: subtract 1.7, multiply by 0.7)
  2. log_energy (standardized: subtract 2.0, multiply by 0.7)
  3. log_pt_rel (standardized: add 4.7, multiply by 0.7)
  4. log_energy_rel (standardized: add 4.7, multiply by 0.7)
  5. dphi (no standardization)
  6. deta (no standardization)
  7. dr (standardized: subtract 0.2, multiply by 4)
- `get_spurion()` - Creates reference vectors for coordinate system
- `TAGGING_FEATURES_PREPROCESSING` - Standardization constants

## Adaptation Notes

If you want to use `embed_tagging_data()` from `embedding.py`, you'll need to replace these LLoCa dependencies:

1. **`utils.get_batch_from_ptr()`** → Simple implementation:
   ```python
   def get_batch_from_ptr(ptr):
       batch = torch.zeros(ptr[-1], dtype=torch.long, device=ptr.device)
       for i in range(len(ptr)-1):
           batch[ptr[i]:ptr[i+1]] = i
       return batch
   ```

2. **`lorentz.lorentz_squarednorm()`** → Simple implementation:
   ```python
   def lorentz_squarednorm(fourmomenta):
       return fourmomenta[..., 0]**2 - (fourmomenta[..., 1:]**2).sum(dim=-1)
   ```

3. **`polar_decomposition.restframe_boost()`** → Can use standard Lorentz boost to rest frame, or skip jet boosting entirely if not needed

## Data Format

The dataset expects `.npz` files with:
- `kinematics_train`, `kinematics_test`, `kinematics_val`: shape `(n_jets, n_particles_max, 4)` - four-momenta (E, px, py, pz)
- `labels_train`, `labels_test`, `labels_val`: shape `(n_jets,)` - boolean labels (True=top, False=QCD)

## Summary

**Keep these 3 files:**
1. `experiments/tagging/dataset.py`
2. `experiments/hep.py`  
3. `experiments/tagging/embedding.py` (with minor adaptations)

**Everything else can be removed** - you'll implement your own training loop, models, and configuration system.

