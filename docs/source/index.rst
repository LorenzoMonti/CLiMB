CLustering In Multiphase Boundaries (CLiMB)
===========================================

A versatile two-phase clustering algorithm designed for datasets with both known and exploratory components.

Features
--------

* **Two-Phase Clustering**: Combines constrained clustering with exploratory clustering to identify both known and novel patterns.
* **Density-Aware**: Uses local density estimation to intelligently filter and assign points.
* **Flexible Exploratory Phase**: Supports multiple clustering algorithms (DBSCAN, HDBSCAN, OPTICS) through a strategy pattern.
* **Visualization Tools**: Built-in 2D and 3D visualization capabilities for cluster analysis.
* **Parameter Tuning**: Builder pattern for flexible parameter adjustment.
* **Customizable Distance Metrics**: Supports various distance metrics such as Euclidean, Mahalanobis, and custom metrics.
* **Advanced Seed Points**: Ability to initialize clustering with known seed points provided in a dictionary structure.

Installation
------------

Install via pip:

.. code-block:: bash

   pip install climb-astro

Or install from source:

.. code-block:: bash

   git clone https://github.com/LorenzoMonti/CLiMB.git
   cd CLiMB
   pip install -e .

Quick Start
-----------

Here is a basic example of how to use CLiMB with synthetic data:

.. code-block:: python

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
       tuple(seed_points_scaled[0]): [tuple(X_scaled[y == 0][0]), tuple(X_scaled[y == 0][1])],
       tuple(seed_points_scaled[1]): [tuple(X_scaled[y == 1][0])],
       tuple(seed_points_scaled[2]): [],
       tuple(seed_points_scaled[3]): [tuple(X_scaled[y == 3][0]), tuple(X_scaled[y == 3][1]), tuple(X_scaled[y == 3][2])]
   }

   # Initialize and fit CLiMB
   climb = CLiMB(
       constrained_clusters=4,
       seed_points=seed_dict_scaled,
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
   # Note: visualization requires dimensionality reduction if features > 3
   climb.inverse_transform(scaler)
   # fig = climb.plot_comprehensive_3d(save_path="./3d")
   # fig2 = climb.plot_comprehensive_2d(save_path="./2d")


API Reference
-------------

Here you can find complete documentation for all classes and methods.

.. toctree::
   :maxdepth: 2

   CLiMB


How It Works
------------

CLiMB operates in two phases:

1. **Constrained Phase (KBound)**: A modified K-means that:
   
   * Uses seed points to guide initial clustering.
   * Applies density and distance constraints.
   * Prevents centroids from drifting too far using radial thresholds.
   * Supports customizable distance metrics.
   * Handles advanced seed points via a dictionary structure.

2. **Exploratory Phase**: Uses density-based clustering methods to discover patterns in points not assigned during the first phase.


Advanced Usage
--------------

Using Different Exploratory Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from CLiMB.core.CLiMB import CLiMB
   from CLiMB.exploratory.HDBSCANExploratory import HDBSCANExploratory

   # Create HDBSCAN exploratory algorithm
   hdbscan = HDBSCANExploratory(min_cluster_size=5, min_samples=3)

   # Use it with CLIMB
   climb = CLiMB(
       constrained_clusters=3,
       exploratory_algorithm=hdbscan
   )

Parameter Tuning with Builder Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   climb = CLiMB()
   climb.set_density(0.3) \
        .set_distance(2.5) \
        .set_radial(1.0) \
        .set_convergence(0.1)

Using Custom Distance Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use distance metrics other than Euclidean, you can use the ``distance_metric`` and ``metric_params`` parameters.

**Example with Mahalanobis Metric:**

.. code-block:: python

   import numpy as np
   from CLiMB.core.KBound import KBound

   # ... (Load or generate your data X) ...

   # Calculate the inverse covariance matrix (VI)
   covariance_matrix = np.cov(X.T)
   inv_covariance_matrix = np.linalg.inv(covariance_matrix)

   kbound = KBound(
       n_clusters=3,
       distance_metric='mahalanobis',
       metric_params={'VI': inv_covariance_matrix}
   )
   kbound.fit(X)

**Example with Custom Metric:**

.. code-block:: python

   import numpy as np
   from scipy.spatial.distance import euclidean
   from CLiMB.core.KBound import KBound

   def custom_distance(u, v):
       # Example: weighted Euclidean distance
       weight = np.array([2, 1, 1]) 
       return euclidean(u * weight, v * weight)

   kbound = KBound(
       n_clusters=3,
       distance_metric='custom',
       metric_params={'func': custom_distance}
   )
   kbound.fit(X)