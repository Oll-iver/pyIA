import pytest
import numpy as np
import scipy.sparse as sp
from pyia import Interval, midrad
from sparse import SparseIntervalMatrix
from solvers import verifylss, sparselss, verifynlss
from ad import Gradient, initvar
import ad  # Imported to check for missing attributes
from scipy.linalg import hilbert

# --- Helper for Rigorous Checks ---
def assert_inclusion(true_val, interval_result):

    if isinstance(interval_result, Interval):
        # Handle Interval scalars/vectors
        lower = interval_result.inf
        upper = interval_result.sup
    elif isinstance(interval_result, Gradient):
        # Handle AD Gradient objects (check value component)
        if isinstance(interval_result.x, Interval):
            lower = interval_result.x.inf
            upper = interval_result.x.sup
        else:
            lower = interval_result.x
            upper = interval_result.x
    else:
        # Fallback for raw floats
        lower = interval_result
        upper = interval_result

    # Check bounds
    in_lower = np.all(lower <= true_val + 1e-14) # Small epsilon for float jitter
    in_upper = np.all(true_val <= upper + 1e-14)
    
    if not (in_lower and in_upper):
        pytest.fail(
            f"VIOLATION OF INCLUSION PROPERTY!\n"
            f"True Value: {true_val}\n"
            f"Interval:   [{np.min(lower)}, {np.max(upper)}]\n"
            f"Failed indices: {np.where(~(lower <= true_val) | ~(true_val <= upper))}"
        )

# ==========================================
# 1. Interval Arithmetic Tests (pyia.py)
# ==========================================

def test_interval_basic_arithmetic():
    """Test fundamental commutative and associative properties with inclusion."""
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 4.0)
    
    # Addition
    c = a + b
    assert_inclusion(1.0 + 3.0, c) # Lower bound check
    assert_inclusion(2.0 + 4.0, c) # Upper bound check
    
    # Multiplication (Mixed signs to trigger min/max logic)
    d = Interval(-2.0, 1.0)
    e = Interval(3.0, 5.0)
    prod = d * e
    # True range: [-10, 5]
    assert np.isclose(prod.inf, -10.0)
    assert np.isclose(prod.sup, 5.0)

def test_interval_transcendental_functions():
    """Test sin, cos, exp, log inclusion."""
    val = 1.5
    x = Interval(val) # Point interval
    
    # Exp
    assert_inclusion(np.exp(val), x.exp())
    
    # Log
    assert_inclusion(np.log(val), x.log())
    
    # Sin (Check critical points containment)
    # Interval [0, pi] should contain 0.0 and 1.0
    half_circle = Interval(0.0, np.pi)
    res = half_circle.sin()
    assert_inclusion(1.0, res)  # Peak
    assert_inclusion(0.0, res)  # Endpoints
    assert res.sup >= 1.0       # Should capture peak exactly or round up
    
    # Cos
    res_cos = half_circle.cos()
    assert_inclusion(-1.0, res_cos) # cos(pi)
    assert_inclusion(1.0, res_cos)  # cos(0)

# ==========================================
# 2. Automatic Differentiation Tests (ad.py)
# ==========================================

def test_ad_chain_rule_basic():
    """Test gradients for f(x) = x^2 + 3x."""
    val = 2.0
    x = Gradient(val, 1.0) # dx = 1
    
    # f(x) = x*x + 3*x
    f = x*x + Gradient(3.0)*x
    
    true_val = val**2 + 3*val     # 10
    true_grad = 2*val + 3         # 7
    
    assert np.isclose(f.x, true_val)
    assert np.isclose(f.dx, true_grad)

def test_ad_exp_chain():
    """Test f(x) = exp(x*2)."""
    val = 1.0
    x = Gradient(val, 1.0)
    
    f = (x * Gradient(2.0)).exp()
    
    true_val = np.exp(val * 2)
    true_grad = 2 * np.exp(val * 2)
    
    assert np.isclose(f.x, true_val)
    assert np.isclose(f.dx, true_grad)

def test_ad_missing_features():
    """
    CRITICAL CHECK: Does ad.py support sin/cos/log?
    The current file provided ONLY has exp. This test WILL FAIL 
    until you implement them, alerting you to the missing features.
    """
    x = Gradient(0.5, 1.0)
    
    try:
        y = x.sin()
    except AttributeError:
        pytest.fail("CRITICAL: 'sin' method missing in ad.py Gradient class!")
        
    try:
        y = x.cos()
    except AttributeError:
        pytest.fail("CRITICAL: 'cos' method missing in ad.py Gradient class!")
        
    try:
        y = x.log()
    except AttributeError:
        pytest.fail("CRITICAL: 'log' method missing in ad.py Gradient class!")

def test_ad_interval_mixed():
    """Test AD works with Intervals (Validation + Derivatives)."""
    # f(x) = x * x
    # x = [1, 2]
    # f(x) = [1, 4]
    # f'(x) = 2x = [2, 4]
    
    i_val = Interval(1.0, 2.0)
    x = Gradient(i_val, Interval(1.0)) # dx = 1 (Interval)
    
    f = x * x
    
    assert_inclusion(1.0, f.x) # f(1)
    assert_inclusion(4.0, f.x) # f(2)
    
    assert_inclusion(2.0, f.dx) # f'(1)
    assert_inclusion(4.0, f.dx) # f'(2)

def test_ad_power_operator():
    val = 3.0
    x = Gradient(val, 1.0) # dx=1
    
    # f(x) = x^3
    f = x ** 3
    
    assert np.isclose(f.x, 27.0)   # 3^3
    assert np.isclose(f.dx, 27.0)  # 3*x^2 = 3*9 = 27


# ==========================================
# 3. Solver Tests (Dense & Sparse)
# ==========================================

def test_dense_solver_Ax_b():
    """
    Solve Ax=b for DENSE Interval Matrix.
    Uses a random matrix with known solution.
    """
    N = 50
    np.random.seed(42)
    
    # 1. Generate Float Data
    A_mid = np.random.rand(N, N) + np.eye(N)*N # Diagonally dominant to ensure non-singular
    x_true = np.ones(N)
    b_mid = A_mid @ x_true
    
    # 2. Add Uncertainty
    A_rad = np.full((N, N), 1e-5)
    b_rad = np.full(N, 1e-5)
    
    A = midrad(A_mid, A_rad)
    b = midrad(b_mid, b_rad)
    
    # 3. Solve
    x_interval = verifylss(A, b)
    
    assert x_interval is not None, "Dense solver failed to converge"
    
    # 4. Rigorous Check
    # Since we added radius to A and b, the true 'x' (for the midpoint system)
    # MUST be inside the computed interval.
    assert_inclusion(x_true, x_interval)
    
    # Check width is reasonable (tightness)
    width = np.max(x_interval.sup - x_interval.inf)
    assert width < 1e-3, f"Interval solution too wide: {width}"

def test_dense_solver_AX_B_Matrix_Solve():
    """
    Solve AX=B where B is a MATRIX (Interval Matrix).
    Specifically, we compute A^-1 by solving AX = I.
    """
    N = 10
    np.random.seed(101)
    
    # A is a random matrix
    A_mid = np.random.rand(N, N) + np.eye(N)*5
    A = Interval(A_mid) # Point matrix
    
    # B is Identity (Point matrix)
    B = Interval(np.eye(N))
    
    # Solve AX = I  => X should be A_inv
    X_interval = verifylss(A, B)
    
    assert X_interval is not None
    
    # Check against numpy inverse
    inv_true = np.linalg.inv(A_mid)
    assert_inclusion(inv_true, X_interval)
    
    print(f"\nMatrix Inversion Width Max: {np.max(X_interval.sup - X_interval.inf)}")

def test_sparse_solver_Ax_b():
    """
    Solve Ax=b using the SPARSE solver.
    Uses 1D Laplacian (Tridiagonal) which is Symmetric Positive Definite.
    """
    N = 100
    
    # 1. Setup 1D Laplacian: [4, -1, -1] (Strongly diag dominant for Gershgorin)
    diagonals = [np.full(N, 4.0), np.full(N-1, -1.0), np.full(N-1, -1.0)]
    offsets = [0, -1, 1]
    
    mid_A = sp.diags(diagonals, offsets, format='csr')
    rad_A = sp.diags([d * 1e-10 for d in diagonals], offsets, format='csr') # Tiny radius
    rad_A = np.abs(rad_A)
    
    A_sparse = SparseIntervalMatrix(mid_A, rad_A)
    
    # 2. RHS
    x_true = np.ones(N)
    b_mid = mid_A @ x_true
    b = midrad(b_mid, 1e-10)
    
    # 3. Solve
    x_sol = sparselss(A_sparse, b)
    
    assert x_sol is not None, "Sparse solver returned None"
    
    # 4. Inclusion Check
    assert_inclusion(x_true, x_sol)
    
    # 5. Width check
    width = np.max(x_sol.sup - x_sol.inf)
    assert width < 1e-5, f"Sparse solution too wide: {width}"


# ==========================================
# 4. Stress Tests
# ==========================================

    
def test_nonlinear_sphere_intersection():
    """
    Solves Intersection of Sphere and Paraboloid.
    Stresses: AD Vector System + Power Operator (**)
    """
    def sphere_f(x_grad):
        x, y, z = x_grad[0], x_grad[1], x_grad[2]
        
        # 1. Sphere: x^2 + y^2 + z^2 - 1 = 0
        f1 = x**2 + y**2 + z**2 - 1.0
        
        # 2. Paraboloid: x^2 + y^2 - z = 0
        f2 = x**2 + y**2 - z
        
        # 3. Plane: x - y = 0
        f3 = x - y
        
        # We check if the result is an Interval (has .inf attribute)
        if hasattr(f1.x, 'inf'):
            # Interval Mode: We must construct a vector Interval from the scalar Intervals.
            # We use .item() to safely extract the float from the 0-d numpy array inside the scalar.
            infs = np.array([f1.x.inf.item(), f2.x.inf.item(), f3.x.inf.item()])
            sups = np.array([f1.x.sup.item(), f2.x.sup.item(), f3.x.sup.item()])
            vals = Interval(infs, sups)
        else:
            # Float Mode (Newton): Standard numpy array construction works fine
            vals = np.array([f1.x, f2.x, f3.x], dtype=np.float64)
            
        # Stack Jacobian rows
        jac = np.vstack([f1.dx, f2.dx, f3.dx])
        
        return Gradient(vals, jac)
    # Intersection is at z = (sqrt(5)-1)/2 approx 0.618
    # x^2 = z/2 approx 0.309 -> x approx 0.55
    xs = [0.5, 0.5, 0.5]
    
    sol = verifynlss(sphere_f, xs)
    
    assert sol is not None, "Sphere intersection failed to converge"
    
    # Check Analytical Truth
    z_true = (np.sqrt(5) - 1) / 2
    x_true = np.sqrt(z_true / 2)
    
    # Assert x and y are the same
    assert_inclusion(x_true, sol[0])
    assert_inclusion(x_true, sol[1])
    assert_inclusion(z_true, sol[2])
    
    print(f"\nSphere Test Width: {np.max(sol.sup - sol.inf):.2e}")

def test_matrix_inversion_hilbert():
    """
    Solves AX = I (Matrix Inversion) for Hilbert Matrix.
    Stresses: Interval Matrix Multiplication (@) and Linear Solver logic.
    """
    N = 6
    H = hilbert(N)
    A = Interval(H)
    I = Interval(np.eye(N))
    
    # Solve AX = I
    A_inv = verifylss(A, I)
    
    assert A_inv is not None, "Hilbert inversion failed"
    
    # Check: A * A_inv should contain Identity
    Product = A @ A_inv
    
    # The diagonal of Product should contain 1.0
    diag_inf = Product.inf.diagonal()
    diag_sup = Product.sup.diagonal()
    
    assert np.all(diag_inf <= 1.0)
    assert np.all(diag_sup >= 1.0)
    
    print(f"\nHilbert Inversion (N={N}) Verified.")