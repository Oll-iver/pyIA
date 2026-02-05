import numpy as np
from pyia import Interval, midrad
import scipy.sparse.linalg as spla
from sparse import SparseIntervalMatrix
from ad import initvar


def verifylss(A, b):
    """
    Verified Linear System Solver (Algorithm 5.1 from INTLAB - INTERVAL LABORATORY by Rump with Refinement)
    Solves Ax = b. Works for vectors (Ax=b) and matrices (AX=I).
    """
    if not isinstance(A, Interval): A = Interval(A)
    if not isinstance(b, Interval): b = Interval(b)

    # --- STEP 1: Calculate Approximate Inverse with Newton Refinement ---
    try:
        # Initial floating-point guess
        R_approx = np.linalg.inv(A.mid)

        # Newton Refinement: R = R + R(I - AR)
        # This makes R much closer to A^-1, shrinking the error term C later.
        n = A.shape[0]
        I_mat = np.eye(n)
        Residual = I_mat - (A.mid @ R_approx)
        R_approx = R_approx + (R_approx @ Residual)
        
    except np.linalg.LinAlgError:
        print("ERROR: Matrix is singular (midpoint).")
        return None
    
    # --- STEP 2: Setup Residuals (Standard Algorithm 5.1) ---
    # Approximate solution xs = R * mid(b)
    xs = R_approx @ b.mid
    xs_interval = Interval(xs)
    
    # Residual Z = R * (b - A * xs)
    Res = b - (A @ xs_interval) 
    R_int = Interval(R_approx)
    Z = R_int @ Res
    
    # Preconditioner Residual C = I - R*A
    I = Interval(np.eye(n))
    C = I - (R_int @ A)
    
    # Heuristic check for convergence
    norm_C = np.linalg.norm(np.abs(C.mid), ord=np.inf)
    if norm_C >= 1.0:
        print(f"Warning: Iteration matrix norm {norm_C:.2f} >= 1. Unlikely to converge.")

    # --- STEP 3: Optimised Krawczyk Loop ---
    Y = Z.copy()
    k = 0
    kmax = 100 
    ready = False
    
    hull_neg1_1 = Interval(np.full(Y.shape, -1.0), np.full(Y.shape, 1.0))
    realmin = 1e-300 
    small_eps = midrad(0, 10 * realmin) 

    print("Starting Optimised Krawczyk iteration...")
    while not ready and k < kmax:
        k += 1
        
        # 1. Epsilon Inflation (Explode slightly to find inclusion)
        E = (hull_neg1_1 * (Y.rad * 0.1)) + small_eps
        X = Y + E
        
        # 2. Contraction Step
        # Y_new = Z + C * X
        Y_new = Z + (C @ X)
        
        # 3. Intersection Step (The Squeeze)
        # We know the solution must be inside X (by construction of inclusion)
        # AND it must be inside Y_new (by the fixed point property).
        # Therefore, it must be in the intersection.
        Y_intersect = Y_new.intersect(X)
        
        # Check for empty intersection (NaN) - implies no solution
        if np.any(np.isnan(Y_intersect.inf)):
             # If intersection failed, just keep Y_new 
             pass 
        else:
             Y_new = Y_intersect

        # 4. Convergence Check
        # If the new squeezed Y is strictly inside the expanded X, we are done.
        if Y_new.in0(X):
            ready = True
            
        Y = Y_new

    if ready:
        print(f"Converged in {k} iterations.")
        return xs_interval + Y
    else:
        print(f"Failed to converge after {k} iterations.")
        return None


def verifynlss(f, xs):
    """
    Verified Nonlinear System Solver (Algorithm 5.7)
    
    Args:
        f: A function f(x) written using standard operators. 
           Must accept and return Gradient objects.
        xs: Initial guess (float list/array).
        
    Returns:
        Interval vector containing the unique root, or None.
    """
    xs = np.array(xs, dtype=np.float64)
    n = len(xs)
    
    # We iterate purely in floating point first to get a very good xs
    k = 0
    kmax = 20
    xs_old = xs.copy()
    
    while k < kmax:
        k += 1
        xs_old = xs.copy()
        
        # 1. Initialize AD variables at current point xs
        #    This automatically sets derivatives to Identity matrix
        x_ad = initvar(xs)
        
        # 2. Evaluate function. y will contain value f(x) and Jacobian f'(x)
        y = f(x_ad)
        
        # 3. Newton Step: xs_new = xs - J^-1 * f(xs)
        #    y.dx is the Jacobian matrix (float)
        #    y.x  is the function value vector (float)
        try:
            delta = np.linalg.solve(y.dx, y.x) # Faster/stable than inv()
            xs = xs - delta
        except np.linalg.LinAlgError:
            print("Singular Jacobian in Newton phase.")
            return None
            
        # Check convergence
        if np.linalg.norm(xs - xs_old) < 1e-12 * np.linalg.norm(xs):
            break
            
    print(f"Newton converged to approx root in {k} iterations.")
    
    # Now we try to prove a unique root exists near xs
    
    # 1. Compute approximate inverse R of the Jacobian at the solution
    #    We assume the last y.dx from Newton phase is good enough
    try:
        R = np.linalg.inv(y.dx)
    except np.linalg.LinAlgError:
        return None
        
    # 2. Calculate Residual Z = -R * f(xs)
    #    We evaluate f at the *Interval* point xs (width=0) to catch rounding errors
    xs_int = Interval(xs)
    
    val_check = f(initvar(xs_int)) 
    Z = -Interval(R) @ val_check.x
    X = Z # Initial center
    
    # Hull of [-1, 1] for inflation
    hull_neg1_1 = Interval(np.full(n, -1.0), np.full(n, 1.0))
    realmin = 1e-300
    small_eps = midrad(0, 10 * realmin)
    
    ready = False
    k = 0
    k_verif_max = 15
    Y = X # Initial Y
    
    while not ready and k < k_verif_max:
        k += 1
        
        # Epsilon Inflation: Blow up Y slightly to try and capture the solution
        E = (hull_neg1_1 * (Y.rad * 0.1)) + small_eps
        X = Y + E  # This is the "Candidate Interval" centered at xs
        
        # 4. Krawczyk Operator: K(X) = Z + (I - R*J(X)) * X
        #    We need the Jacobian J evaluated over the Interval X.
        #    This captures ALL slopes in the region.
        
        x_ad_interval = initvar(xs_int + X) #
        y_interval = f(x_ad_interval)       # Evaluate to get Interval Jacobian
        
        # C = I - R * Jacobian(X)
        I = Interval(np.eye(n))
        C = I - (Interval(R) @ y_interval.dx)
        
        # Y_new = Z + C * X
        Y_new = Z + (C @ X)
        
        # 5. Check Intersection / Convergence
        if Y_new.in0(X):
            ready = True
            # Optional: Intersect to tighten
            Y = Y_new.intersect(X)
        else:
            # Did not converge yet, update Y for next inflation
            Y = Y_new

    if ready:
        print(f"Verified unique solution in {k} steps.")
        return xs_int + Y
    else:
        print("Verification failed.")
        return None
    
def sparselss(A: SparseIntervalMatrix, b: Interval):
    """
    Verified Sparse Linear Solver (Algorithm 5.5 adapted).
    Solves Ax=b where A is Symmetric Positive Definite.
    """
    n = A.shape[0]
    
    # 1. Approximate Solution (Floating Point)
    xs = spla.spsolve(A.mid, b.mid)

    # 2. Sigma Min Estimation 
    
    sigma = None
    
    # Calculate Gershgorin Lower Bound:
    # LB_i = |A_ii| - sum(|A_ij| for j != i)
    # This is equivalent to 2*|A_ii| - sum(|Row_i|)
    
    # Get diagonal elements
    diag = A.mid.diagonal()
    abs_diag = np.abs(diag)
    
    # Calculate sum of absolute values for each row
    # (Scipy sparse matrices handle this efficiently)
    row_sums = np.array(np.abs(A.mid).sum(axis=1)).flatten()
    
    # Gershgorin bound vector
    gersh_bounds = 2 * abs_diag - row_sums
    min_gersh = np.min(gersh_bounds)
    
    if min_gersh > 0:
        # Case 1: Matrix is Strictly Diagonally Dominant
        sigma = min_gersh
    else:
        return None

    if sigma <= 0:
        return None 

    # 3. Residual
    Ax = A @ xs
    Residual = b - Ax

    # 4. Error Bound
    res_max = np.max(np.maximum(np.abs(Residual.inf), np.abs(Residual.sup)))
    err_bound = res_max / sigma
    
    return Interval(xs - err_bound, xs + err_bound)