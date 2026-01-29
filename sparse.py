import numpy as np
import scipy.sparse as sp
from pyia import Interval
from pyia_kernel import RoundingMode

class SparseIntervalMatrix:
    """
    Represents a sparse interval matrix A = [mid-rad, mid+rad].
    Optimised for memory efficiency using scipy.sparse.csr_matrix.
    """
    def __init__(self, mid_matrix, rad_matrix):
        # Convert to Compressed Sparse Row (CSR) for fast arithmetic
        self.mid = sp.csr_matrix(mid_matrix)
        self.rad = sp.csr_matrix(rad_matrix)
        self.shape = self.mid.shape

    def __matmul__(self, other):
        """
        Implements verified matrix-vector multiplication: y = A @ x
        Handles: A (Sparse) @ x (Dense Vector) -> y (Interval Vector)
        """
        # 1.  'other' to mid/rad arrays
        if isinstance(other, Interval):
            x_mid, x_rad = other.mid, other.rad
            x_abs = np.abs(x_mid) + x_rad
        else:
            # Assume 'other' is a standard numpy array (point vector)
            x_mid = np.asarray(other)
            x_abs = np.abs(x_mid)
            x_rad = np.zeros_like(x_mid)

        # 2. Compute Radius (Round Up)
        # Formula: rad(y) = rad(A)*|x| + |mid(A)|*rad(x) + float_error
        with RoundingMode(1): # Force Round Up
            # Term 1: Uncertainty in A times magnitude of x
            # (A.rad is non-negative, so dot product is safe)
            r_y = self.rad.dot(x_abs + x_rad)
            
            # Term 2: Magnitude of A times uncertainty in x (if x is interval)
            if np.any(x_rad > 0):
                abs_mid = np.abs(self.mid) 
                r_y += abs_mid.dot(x_rad)
                
            # Term 3: Floating point rounding error of the midpoint product
            # Error ~ |mid(A)| * |mid(x)| * machine_epsilon
            abs_mid_A = np.abs(self.mid)
            float_err = abs_mid_A.dot(np.abs(x_mid)) * 2.22e-16 * 2.0 # 2.0 safety factor
            
            total_rad = r_y + float_err

        # 3. Compute Midpoint (Standard Float)
        y_approx = self.mid.dot(x_mid)

        # 4. Return Interval
        return Interval(y_approx) + Interval(-total_rad, total_rad)