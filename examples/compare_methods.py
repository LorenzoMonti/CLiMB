import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

# Import the CLiMB algorithm and exploratory strategies
from CLiMB.core.CLiMB import CLiMB
from CLiMB.exploratory.HDBSCANExploratory import HDBSCANExploratory 
from CLiMB.exploratory.DBSCANExploratory import DBSCANExploratory
from CLiMB.exploratory.OPTICSExploratory import OPTICSExploratory

def compare_exploratory_methods():
    """
    Compare different exploratory methods on the same dataset
    """
    print("Comparing different exploratory methods...")
    
    # Generate data with both convex and non-convex clusters
    X_blobs, y_blobs = make_blobs(
        n_samples=200, 
        centers=3,
        n_features=3,
        random_state=42,
        cluster_std=0.8
    )
    
    X_moons, y_moons = make_moons(n_samples=150, noise=0.08, random_state=42)
    X_moons = np.column_stack((
        X_moons, 
        np.random.normal(0, 0.1, size=X_moons.shape[0])
    ))
    
    X_combined = np.vstack([X_blobs, X_moons])
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Create seed points for known clusters
    seed_points = np.array([
        X_blobs[y_blobs == label].mean(axis=0) for label in np.unique(y_blobs)
    ])
    seed_points_scaled = scaler.transform(seed_points)
    
    # Create different exploratory algorithms
    exploratory_methods = {
        "DBSCAN": DBSCANExploratory(eps=0.4, min_samples=5),
        "HDBSCAN": HDBSCANExploratory(min_cluster_size=5, min_samples=3),
        "OPTICS": OPTICSExploratory(min_samples=5)
    }
    
    results = {}
    
    # Run CLiMB with each exploratory method
    for name, method in exploratory_methods.items():
        print(f"Testing with {name}...")
        
        climb = CLiMB(
            constrained_clusters=3,
            seed_points=seed_points_scaled,
            density_threshold=0.3,
            distance_threshold=2.5,
            exploratory_algorithm=method
        )
        
        climb.fit(X_scaled)
        labels = climb.get_labels()
        
        # Store results
        results[name] = {
            "labels": labels,
            "constrained_points": len(X_scaled) - len(climb.unassigned_points),
            "exploratory_points": len(climb.unassigned_points),
            "total_clusters": len(np.unique(labels))
        }
        
        # Visualization
        climb.inverse_transform(scaler)
        fig = climb.plot_comprehensive_2d(
            dimensions=(0, 1),
            axis_labels=['Feature 1', 'Feature 2']
        )
        plt.savefig(f'comparison_{name}.png')
        plt.close(fig)
    
    # Print comparison
    print("\nComparison Results:")
    print("-" * 60)
    print(f"{'Method':<10} | {'Constrained %':<15} | {'Exploratory %':<15} | {'Total Clusters':<15}")
    print("-" * 60)
    
    for name, result in results.items():
        constrained_pct = result["constrained_points"] / len(X_scaled) * 100
        exploratory_pct = result["exploratory_points"] / len(X_scaled) * 100
        
        print(f"{name:<10} | {constrained_pct:>13.1f}% | {exploratory_pct:>13.1f}% | {result['total_clusters']:>15}")

if __name__ == "__main__":      
    compare_exploratory_methods()