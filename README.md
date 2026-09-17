<p align="center"> <img src="imgs/icon.png" alt="CLiMB logo" width="300"/> </p>

# CLustering In Multiphase Boundaries (CLiMB)
A versatile two-phase clustering algorithm designed for datasets with both known and exploratory components.

## Features

- **Two-Phase Clustering**: Combines constrained clustering with exploratory clustering to identify both known and novel patterns.
- **Density-Aware**: Uses local density estimation to intelligently filter and assign points.
- **Flexible Exploratory Phase**: Supports multiple clustering algorithms (DBSCAN, HDBSCAN, OPTICS) through a strategy pattern.
- **Visualization Tools**: Built-in 2D and 3D visualization capabilities for cluster analysis.
- **Parameter Tuning**: Builder pattern for flexible parameter adjustment.
- **Customizable Distance Metrics**: Now supports various distance metrics such as Euclidean, Mahalanobis, and custom metrics, offering greater flexibility in distance calculation.
- **Interpretability**: `explain()` reconstructs both phases' decisions in closed form and
  verifies the reconstruction against the model, rather than fitting a surrogate to imitate it.
- **Advanced Seed Points**: Ability to initialize clustering with known seed points provided in a dictionary structure, allowing for more precise control over centroid initialization.

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
from CLiMB.exploratory.DBSCANExploratory import DBSCANExploratory

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

# Example of seed points as a dictionary for more precise control
seed_dict_scaled = {
    tuple(seed_points_scaled[0]): [tuple(X_scaled[y == 0][0]), tuple(X_scaled[y == 0][1])], # Centroid 1 and associated seed points
    tuple(seed_points_scaled[1]): [tuple(X_scaled[y == 1][0])],                             # Centroid 2 and associated seed points
    tuple(seed_points_scaled[2]): [],                                                         # Centroid 3 with no specific seed points
    tuple(seed_points_scaled[3]): [tuple(X_scaled[y == 3][0]), tuple(X_scaled[y == 3][1]), tuple(X_scaled[y == 3][2])] # Centroid 4 and seed points
}

# Initialize and fit CLiMB with Euclidean metric and dictionary seed points
climb = CLiMB(
    constrained_clusters=4,
    seed_points=seed_dict_scaled, # Use the dictionary of seed points
    density_threshold=0.15,
    distance_threshold=2.5,
    radial_threshold=1.2,
    convergence_tolerance=0.05,
    distance_metric='euclidean',
    metric_params=None,
    exploratory_algorithm=DBSCANExploratory(0.5)
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
- `complex_mixed.py`: Handling mixed data with both convex and non-convex clusters
- `mixed_seeds.py`: Handling mixed data with both convex and non-convex clusters with seeds
- `compare_methods.py`: Comparing different exploratory clustering methods

## How It Works

CLiMB operates in two phases:

1. **Constrained Phase (KBound)**: A modified K-means that:
   - Uses seed points to guide initial clustering 
   - Applies density and distance constraints
   - Prevents centroids from drifting too far using radial thresholds
   - Supports customizable distance metrics through the distance_metric and metric_params parameters.
    - Handles advanced seed points via a dictionary structure for more controlled initialization.

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

### Using Custom Distance Metrics
To use distance metrics other than Euclidean, you can use the distance_metric and metric_params parameters in the KBound class.

**Example with Mahalanobis Metric:**

```python
import numpy as np
from CLiMB.core.KBound import KBound

# ... (Load or generate your data X) ...

# Calculate the inverse covariance matrix (VI) for Mahalanobis
covariance_matrix = np.cov(X.T)
inv_covariance_matrix = np.linalg.inv(covariance_matrix)

# Initialize KBound with the Mahalanobis metric and parameters
kbound = KBound(
    n_clusters=3,
    distance_metric='mahalanobis',
    metric_params={'VI': inv_covariance_matrix}
)
kbound.fit(X)
```
If you do not provide metric_params for the Mahalanobis metric, the inverse covariance matrix will be automatically calculated on the input data X during fit.

**Example with Custom Metric:**
```python
import numpy as np
from CLiMB.core.KBound import KBound
from scipy.spatial.distance import euclidean

# Define your custom distance metric
def custom_distance(u, v):
    # Example: weighted Euclidean distance, where the first dimension counts double
    weight = np.array([2, 1, 1]) # Weights for each dimension
    return euclidean(u * weight, v * weight)

# Initialize KBound with the custom metric
kbound = KBound(
    n_clusters=3,
    distance_metric='custom',
    metric_params={'func': custom_distance}
)
kbound.fit(X)
```

### Advanced Usage of Seed Points with Dictionary
In addition to the list of seed points, you can now provide a dictionary to initialize centroids and associate specific seed points with each centroid. This offers more granular control over the initialization of constrained clustering.

**Example of Seed Point Dictionary:**
```python
import numpy as np
from CLiMB.core.KBound import KBound

# ... (Load or generate your data X and define initial centroids and seed points) ...

# Define a dictionary of seed points
# Keys are initial centroids (points), values are lists of seed points associated with that centroid
seed_dict = {
    tuple(initial_centroids[0]): [tuple(seed_point_cluster1_1), tuple(seed_point_cluster1_2)],
    tuple(initial_centroids[1]): [tuple(seed_point_cluster2_1)],
    tuple(initial_centroids[2]): [], # No specific seed points for this centroid
    # ... and so on for all centroids ...
}

# Initialize KBound with the seed point dictionary
kbound = KBound(
    n_clusters=len(initial_centroids),
    seeds=seed_dict
    # ... and so on for the other parameters
)
kbound.fit(X)
```

## Interpretability

CLiMB's decisions are not a black box, so the explanation is not a surrogate.
Phase 1 applies two explicit gates (density, distance) plus seed forcing; DBSCAN
applies a radius rule. Those are rules that can be replayed, so `explain()`
**reconstructs** them in closed form and checks the reconstruction against the
model. SHAP or LIME would fit a second model to imitate the first; here there is
nothing to imitate.

```python
explanation = climb.explain(X_scaled, feature_names=["f1", "f2", "f3"])
```

One row per point: `phase1_*` columns for every point, `phase2_*` for the points
Phase 1 rejected, and `final_label`. Call it *before* `inverse_transform()`,
which rewrites the stored centroids in place; pass `scaler=` to report the
feature columns in original units instead.

### What is verified, and what is not

The package keeps three kinds of output apart rather than letting one borrow
another's credibility.

**Exact reconstructions** — replayed from the fitted model and verified on every
call, raising rather than returning a table that does not describe the model:

| | reconstructs | checked against |
|---|---|---|
| `KBound.decision_path` | gates and labels | `labels_` on non-seed points |
| `KBound.feature_attribution` | additive split of the metric | shares sum to the distance |
| `DBSCANExploratory.explain` | core/border/noise roles | `core_sample_indices_` |
| `OPTICSExploratory.explain` | core distances | `core_distances_` |

Seed points are excluded from the Phase-1 check because their label is copied,
not derived, which would make the check vacuous.

**Reported model state** — exact, but with nothing independent to check it
against. `HDBSCANExploratory.explain` reports membership probability, GLOSH
outlier score and cluster persistence; its labels come from the stability of a
condensed tree and there is no closed form to replay, so `fidelity_` stays
`None` rather than carrying a meaningless 1.0.

**Descriptive statistics** — `CLiMB.explain.descriptive` holds contrastive
signatures and effect sizes. These are computed after the fact, by our rules,
and nothing verifies them. A signature says *these clusters differ on this
feature*; it does not say *this is why the point was assigned*. For that, use
`feature_attribution`, which decomposes the metric the model actually used.

### Why the three algorithms explain differently

`explain()` is abstract on `ExploratoryClusteringBase` and implemented
separately, not generalised. Phase 2's algorithms do not share a decision
structure: DBSCAN assigns roles at one fixed radius, OPTICS refuses to fix a
radius and orders points by reachability, HDBSCAN decides membership by cluster
stability across every scale at once. Common columns would mean inventing a
core/border split for two algorithms that have none.

### Plots

The helpers in `CLiMB.explain` are domain-neutral: axis and colourbar text is
always the caller's, never inferred from the features.

```python
from CLiMB.explain import plot_gate_accounting, plot_margin_map

plot_gate_accounting(explanation, column="phase1_gate")
plot_margin_map(explanation, x="f1", y="f2", value="phase1_distance_margin",
                xlabel="f1 (your units)", ylabel="f2 (your units)")
```

## License

MIT

## Tests Status
![Test Status](https://github.com/LorenzoMonti/CLiMB/actions/workflows/test.yml/badge.svg)

## Citation
