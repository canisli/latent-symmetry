# Kinematic Polynomials

Kinematic polynomials are Lorentz-invariant multiparticle correlators based on Mandelstam invariants. This document describes their implementation in EnergyFlow.

## Mathematical Definition

For a collection of $M$ particles with four-momenta $p_i^\mu$, a kinematic polynomial indexed by a multigraph $G$ is defined as:

$$\text{KP}_G = \sum_{i_1=1}^M \sum_{i_2=1}^M \cdots \sum_{i_N=1}^M \prod_{(k,\ell) \in G} s_{i_k i_\ell}$$

where $s_{ij} = (p_i + p_j)^2$ is the Mandelstam invariant for the pair of particles $i$ and $j$.

### Mandelstam Invariant

The Mandelstam invariant is computed as:

$$s_{ij} = (p_i + p_j)^2 = m_i^2 + m_j^2 + 2 p_i \cdot p_j$$

where:
- $m_i^2 = p_i^\mu p_{i\mu} = E_i^2 - |\vec{p}_i|^2$ is the invariant mass squared
- $p_i \cdot p_j = E_i E_j - \vec{p}_i \cdot \vec{p}_j$ is the Lorentz-invariant dot product

using the $(+,-,-,-)$ metric convention.

### Relation to Energy Flow Polynomials

Kinematic polynomials are a special case of Energy Flow Polynomials (EFPs) with:
- **Energy measure**: $z_i = 1$ (uniform weights for all particles)
- **Angular measure**: $\theta_{ij} = (p_i + p_j)^2$ (Mandelstam invariant, no $\beta$ exponent)

This makes kinematic polynomials fully **Lorentz invariant**, unlike standard EFPs which are only invariant under specific Lorentz subgroups.

## Usage

### Creating a Kinematic Polynomial

Use the `EFP` class with `measure='kinematic'`:

```python
from energyflow import EFP
import numpy as np

# Define a graph by its edges
# Example: triangle graph with vertices 0, 1, 2
triangle = [(0,1), (1,2), (0,2)]

# Create the kinematic polynomial
kp = EFP(triangle, measure='kinematic')
```

### Computing on Events

Events should be provided as arrays of four-momenta in `[E, px, py, pz]` format:

```python
# Event with 3 particles: each row is [E, px, py, pz]
event = np.array([
    [10.0, 3.0, 4.0, 5.0],
    [8.0, 2.0, 3.0, 4.0],
    [5.0, 1.0, 2.0, 2.0]
])

# Compute the kinematic polynomial
result = kp.compute(event)
print(f"KP value: {result}")
```

### Using Hadronic Coordinates

You can also use `[pt, y, phi, m]` (transverse momentum, rapidity, azimuthal angle, mass) coordinates:

```python
kp = EFP([(0,1), (1,2)], measure='kinematic', coords='ptyphim')

# Event in hadronic coordinates: [pt, y, phi] or [pt, y, phi, m]
event_hadr = np.array([
    [50.0, 0.5, 1.2],
    [30.0, -0.3, 2.5],
    [20.0, 0.1, 0.8]
])

result = kp.compute(event_hadr)
```

### Direct Access to the Measure

You can also use the `Measure` class directly to get $z_i$ and $\theta_{ij}$:

```python
from energyflow import Measure

kmeas = Measure('kinematic')
zs, thetas = kmeas.evaluate(event)

print(f"z values: {zs}")  # All ones
print(f"Mandelstam matrix shape: {thetas.shape}")
print(f"s_01 = (p0 + p1)^2 = {thetas[0,1]}")
```

### Batch Computation

For computing on multiple events:

```python
from energyflow import EFPSet

# Create a set of kinematic polynomials
kp_set = EFPSet('d<=4', measure='kinematic')

# Compute on multiple events
events = [np.random.rand(10, 4) for _ in range(100)]
results = kp_set.batch_compute(events)
```

## Example Graphs

| Graph Name | Edges | Physical Interpretation |
|------------|-------|------------------------|
| Line | `[(0,1)]` | Two-particle invariant $\sum_{i,j} s_{ij}$ |
| Triangle | `[(0,1), (1,2), (0,2)]` | Three-particle correlator |
| Square | `[(0,1), (1,2), (2,3), (3,0)]` | Four-particle ring |
| Star | `[(0,1), (0,2), (0,3)]` | Central particle correlations |

## Properties

1. **Lorentz Invariance**: Kinematic polynomials are fully Lorentz invariant since both the Mandelstam invariants $s_{ij}$ and the uniform weights $z_i = 1$ are Lorentz scalars.

2. **No Normalization**: Unlike standard EFPs, kinematic polynomials use $z_i = 1$ without normalization, so the `normed` parameter has no effect.

3. **No Beta Parameter**: The angular measure uses raw $(p_i + p_j)^2$ without any exponent, so the `beta` parameter is ignored.

4. **Massive Particles**: The implementation correctly handles massive particles using the full Mandelstam invariant formula.

## Implementation Details

The kinematic measure is implemented in `energyflow/measure.py` as the `KinematicMeasure` class. Key methods:

- `_mandelstam_matrix(p4s)`: Computes the matrix of all pairwise Mandelstam invariants
- `_kinematic_eval(arg)`: Returns `(zs, thetas)` where `zs` is all ones and `thetas` is the Mandelstam matrix

The computation uses efficient NumPy einsum operations:

```python
# Mass squared: m² = E² - px² - py² - pz²
m2 = np.einsum('ij,ij,j->i', p4s, p4s, metric)

# Dot product matrix: 2 * p_i · p_j  
dot_matrix = 2 * np.einsum('ik,jk,k->ij', p4s, p4s, metric)

# Mandelstam matrix: s_ij = m_i² + m_j² + 2p_i·p_j
thetas = m2[:, np.newaxis] + m2[np.newaxis, :] + dot_matrix
```

where `metric = [1, -1, -1, -1]` is the Minkowski metric.

