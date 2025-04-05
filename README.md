<p align="center"> <img src="imgs/icon.png" alt="CLiMB logo" width="300"/> </p>

# CLustering In Multiphase Boundaries (CLiMB)
A versatile two-phase clustering algorithm designed for datasets with both known and exploratory components.

## Features

- **Two-Phase Clustering**: Combines constrained clustering with exploratory clustering to identify both known and novel patterns.
- **Density-Aware**: Uses local density estimation to intelligently filter and assign points.
- **Flexible Exploratory Phase**: Supports multiple clustering algorithms (DBSCAN, HDBSCAN, OPTICS) through a strategy pattern.
- **Visualization Tools**: Built-in 2D and 3D visualization capabilities for cluster analysis.
- **Parameter Tuning**: Builder pattern for flexible parameter adjustment.

## Installation

```bash
pip install climb-astro
```

Or install from source:

```bash
git clone https://github.com/LorenzoMonti/CLiMB.git
cd CLiMB
pip install -e .
```

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from CLiMB.core.CLiMB import CLiMB

# The number of centers to generate
centers = 4

# Generate synthetic data with 5 dimensions
X, y = make_blobs(n_samples=500, centers=centers, n_features=5, random_state=42)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create seed points (optional)
seed_points = np.array([
    X[y == i].mean(axis=0) for i in range(centers)
])
seed_points_scaled = scaler.transform(seed_points)

# Initialize and fit CLiMB
climb = CLiMB(
    constrained_clusters=4,
    seed_points=seed_points_scaled,
    density_threshold=0.15,
    distance_threshold=2.5,
    radial_threshold=1.2,
    convergence_tolerance=0.05
)
climb.fit(X_scaled)

# Get cluster labels
labels = climb.get_labels()

# Visualize results (only possible in lower dimensions)
climb.inverse_transform(scaler)
fig = climb.plot_comprehensive_3d(save_path="./3d")
fig2 = climb.plot_comprehensive_2d(save_path="./2d")
```

## Examples

See the `examples/` directory for detailed usage examples:

- `simple_example.py`: Basic usage with well-defined clusters
- `mixed_data_example.py`: Handling mixed data with both convex and non-convex clusters
- `compare_methods.py`: Comparing different exploratory clustering methods

## How It Works

CLiMB operates in two phases:

1. **Constrained Phase (KBound)**: A modified K-means that:
   - Uses seed points to guide initial clustering 
   - Applies density and distance constraints
   - Prevents centroids from drifting too far using radial thresholds

2. **Exploratory Phase**: Uses density-based clustering methods to discover patterns in points not assigned during the first phase.

## Use Cases

CLiMB is particularly useful for:

- Datasets with partially known structure
- Astronomical data analysis
- Particle physics clustering
- Pattern discovery in scientific datasets
- Data exploration with prior knowledge

## Advanced Usage

### Using Different Exploratory Algorithms

```python
from CLiMB.core.CLiMB import CLiMB
from CLiMB.exploratory.HDBSCANExploratory import HDBSCANExploratory

# Create HDBSCAN exploratory algorithm
hdbscan = HDBSCANExploratory(min_cluster_size=5, min_samples=3)

# Use it with CLIMB
climb = CLiMB(
    constrained_clusters=3,
    exploratory_algorithm=hdbscan
)
```

### Parameter Tuning with Builder Pattern

```python
climb = CLiMB()
climb.set_density(0.3) \
     .set_distance(2.5) \
     .set_radial(1.0) \
     .set_convergence(0.1)
```

## License

MIT

## Tests Status
![Test Status](https://github.com/LorenzoMonti/CLiMB/actions/workflows/test.yml/badge.svg)

## Citation
