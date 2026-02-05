import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyia import Interval
from solvers import verifylss

def plot_solution_set(A, b, resolution=200):
    """    
    Uses the Oettli-Prager theorem to identify the exact solution set
    and compares it against the interval hull computed by verifylss.
    """
    # Ensure inputs are Intervals
    if not isinstance(A, Interval): A = Interval(A)
    if not isinstance(b, Interval): b = Interval(b)

    n = len(b)
    if n not in [2, 3]:
        print(f"Error: Can only plot 2D or 3D systems (got {n}D).")
        return

    # 1. Compute the Hull (The outer box)
    print("Computing rigorous hull...")
    hull = verifylss(A, b)
    if hull is None:
        print("System is singular or unbounded.")
        return

    # 2. Setup Oettli-Prager Parameters
    Ac = A.mid
    Ar = A.rad
    bc = b.mid
    br = b.rad

    def is_in_solution(pts):
        x = pts.T 
        lhs = np.abs(Ac @ x - bc[:, None]) 
        rhs = Ar @ np.abs(x) + br[:, None]
        # Check if inequality holds for ALL rows (equations)
        return np.all(lhs <= rhs, axis=0)

    # 3. Define Plotting Range (Hull + 20% margin)
    ranges = []
    margin = 0.2
    for i in range(n):
        width = hull[i].rad * 2
        r_min = hull[i].inf - width * margin
        r_max = hull[i].sup + width * margin
        ranges.append((r_min, r_max))

    if n == 2:
        print("Generating 2D Contour Plot...")
        x = np.linspace(ranges[0][0], ranges[0][1], resolution)
        y = np.linspace(ranges[1][0], ranges[1][1], resolution)
        X, Y = np.meshgrid(x, y)
        
        pts = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Check points
        mask = is_in_solution(pts)
        Z = mask.reshape(X.shape).astype(int)

        # Plot
        plt.figure(figsize=(8, 8))
        
        # Exact Solution (Blue Region)
        plt.contourf(X, Y, Z, levels=[0.5, 1.5], colors=['blue'], alpha=0.7)
        plt.contour(X, Y, Z, levels=[0.5], colors=['blue'], linewidths=1)
        
        # Interval Hull (Red Box)
        rect = plt.Rectangle((hull[0].inf, hull[1].inf), 
                             hull[0].rad*2, hull[1].rad*2, 
                             linewidth=2, edgecolor='r', facecolor='none', 
                             linestyle='--', label='Verified Hull')
        plt.gca().add_patch(rect)
        
        plt.title("Exact Solution Set (Blue) vs Hull (Red)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal') # Important to see true shape
        plt.show()

    elif n == 3:
        print("Sampling 3D Point Cloud (this may take a moment)...")
        #Monte Carlo sampling.
        N_samples = 100000
        
        # Uniform sample within the view box
        pts = np.random.uniform(low=[r[0] for r in ranges], 
                                high=[r[1] for r in ranges], 
                                size=(N_samples, 3))
        
        mask = is_in_solution(pts)
        valid_pts = pts[mask]

        if len(valid_pts) == 0:
            print("No solution points found in sampling range.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Exact Solution (Blue Dots)
        ax.scatter(valid_pts[:,0], valid_pts[:,1], valid_pts[:,2], 
                   s=1, c='#3498db', alpha=0.2, label='Exact Set')
        
        # Plot Hull (Red Wireframe Box)
        # Just plotting the bounding limits for clarity
        ax.set_xlim(ranges[0])
        ax.set_ylim(ranges[1])
        ax.set_zlim(ranges[2])
        
        ax.set_title(f"3D Solution Set ({len(valid_pts)} points)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        plt.show()

if __name__ == "__main__":
    # --- 2D Example (Star Shape) ---
    print("--- 2D Visualisation ---")
    A2 = Interval([[3, -1], [-1, 3]]) + Interval(np.full((2,2), -0.5), np.full((2,2), 0.5))
    b2 = Interval([-1, -1], [1, 1])
    plot_solution_set(A2, b2) 

    # --- 3D Example (Polytope Cloud) ---
    print("\n--- 3D Visualisation ---")
    # Define a 3x3 Diagonally Dominant Matrix
    A_mid = np.array([
        [4.0, -1.0, -1.0],
        [-1.0, 4.0, -1.0],
        [-1.0, -1.0, 4.0]
    ])
    # Radius is 0.5 everywhere
    A_rad = np.full((3, 3), 0.5)
    
    A3 = Interval(A_mid) + Interval(-A_rad, A_rad)
    
    # Vector b is [-1, 1] for all components
    b3 = Interval(np.zeros(3)) + Interval(np.full(3, -1.0), np.full(3, 1.0))
    
    print(f"Matrix A (3x3):\n{A3}")
    plot_solution_set(A3, b3)