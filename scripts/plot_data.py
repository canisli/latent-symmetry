#!/usr/bin/env python3
"""Plot the scalar field dataset."""

from so2toy.data import ScalarFieldDataset
import matplotlib.pyplot as plt

dataset = ScalarFieldDataset(n_samples=1000, seed=42)
X, y = dataset.get_numpy()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20)
plt.colorbar(label='Scalar field value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scalar Field Dataset')
plt.gca().set_aspect('equal')
plt.show()
