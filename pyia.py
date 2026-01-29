import numpy as np
from pyia_kernel import setround

#We'll use a cache of powers of 10s in floating point arithmetic to make multiplication more accurate and faster. 
_POWER_10_CACHE = {}

class Interval:
    def __init__(self, value, sup=None):
        """
        Initialise an interval.
        Constructor that handles scalars, arrays, existing Intervals, STRINGS, 
        and mixed lists/arrays of Interval objects/floats.
        """
        # Case 0: Input is a string
        if isinstance(value, str):
            res = str2interval(value)
            self.inf = np.array(res.inf, dtype=np.float64)
            self.sup = np.array(res.sup, dtype=np.float64)
            return

        # Case 1: Input is already a single Interval (Copy constructor)
        if isinstance(value, Interval):
            self.inf = value.inf.copy()
            self.sup = value.sup.copy()
            return

        # Case 2: Input is a list/array (Unpacking mixed content)
        if isinstance(value, (list, tuple, np.ndarray)):
            temp_arr = np.array(value)
            # Check if it contains objects (likely Intervals mixed with floats)
            if temp_arr.dtype == object:
                #Check for .inf, otherwise assume it's a scalar number
                vec_inf = np.vectorize(lambda x: x.inf if hasattr(x, 'inf') else x)
                vec_sup = np.vectorize(lambda x: x.sup if hasattr(x, 'sup') else x)
                
                self.inf = vec_inf(temp_arr).astype(np.float64)
                self.sup = vec_sup(temp_arr).astype(np.float64)
                return

        # Case 3: Standard construction (Scalars/Arrays of numbers)
        if sup is None:
            self.inf = np.array(value, dtype=np.float64)
            self.sup = np.array(value, dtype=np.float64)
        else:
            self.inf = np.array(value, dtype=np.float64)
            self.sup = np.array(sup, dtype=np.float64)
            
        # Error catching
        if np.any(self.inf > self.sup):
            raise ValueError("Invalid interval: inf > sup")

    @property #midpoint of an interval
    def mid(self):
        return 0.5 * (self.inf + self.sup)

    @property #radius of an interval
    def rad(self):
        return 0.5 * (self.sup - self.inf)
    
    @property
    def shape(self):
        return self.inf.shape

    def copy(self):
        return Interval(self.inf.copy(), self.sup.copy())

    def __len__(self):
        return len(self.inf)

    def __repr__(self):
        with np.printoptions(precision=4):
            return f"Interval( \n  inf={self.inf}, \n  sup={self.sup} \n)"

    # --- Arithmetic Operations ---

    def __add__(self, other):
        """
        Here and in many other places in this file we'll use setround.
        setround(-1) rounds DOWN. It is a requirement that no matter what, we round the lower bound down and
        the upper bound up, otherwise we run the risk of rounding the lower bound up and losing values that 
        should be included in the interval. setround(1) dies the opposite. setround(0) is default Python.
        setround is a permanent change and so every time we use it we have to re-set setround(0). 
        """
        if not isinstance(other, Interval):
            other = Interval(other)
        setround(-1) # Down
        lo = self.inf + other.inf
        
        setround(1)  # Up
        hi = self.sup + other.sup
        
        setround(0)
        return Interval(lo, hi)
    
    def __radd__(self, other):
        """Reflected add: float + Interval"""
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        setround(-1)
        lo = self.inf - other.sup  
        setround(1)
        hi = self.sup - other.inf
        setround(0)
        return Interval(lo, hi)
    
    def __rsub__(self, other):
        """Reflected sub: float - Interval"""
        # Convert 'other' (float) to Interval, then subtract 'self'
        if not isinstance(other, Interval):
            other = Interval(other)
        return other.__sub__(self)

    def __neg__(self):
        return Interval(-self.sup, -self.inf)

    def __mul__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)

        # 2. Interval * Interval (Elementwise)
        # Note: This is NOT Matrix Multiplication. See __matmul__ for that.
        setround(-1)
        # 4 combinations for lower bound
        p1 = self.inf * other.inf
        p2 = self.inf * other.sup
        p3 = self.sup * other.inf
        p4 = self.sup * other.sup
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))

        setround(1)
        # 4 combinations for upper bound
        p1 = self.inf * other.inf
        p2 = self.inf * other.sup
        p3 = self.sup * other.inf
        p4 = self.sup * other.sup
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        
        setround(0)
        return Interval(lo, hi)

    def __rmul__(self, other):
        # Commutative elementwise multiplication
        return self.__mul__(other)

    def __matmul__(self, other):
        """
        Implementation of Algorithm 2.7 (Page 5 of INTLAB - INTERVAL LABORATRY by Rump)
        Interval Matrix Multiplication using Midpoint-Radius Arithmetic.
        """
        if not isinstance(other, Interval):
            other = Interval(other)

        # 1. Midpoints and Radii
        Amid = self.mid
        Arad = self.rad
        Bmid = other.mid
        Brad = other.rad

        # 2. Approximate products
        setround(-1)
        C1 = Amid @ Bmid
        
        setround(1)
        C2 = Amid @ Bmid

        # 3. Correction Terms
        setround(0) 
        Cmid = C1 + 0.5 * (C2 - C1)

        setround(1) # Radius must always round up
        term1 = Cmid - C1
        term2 = Arad @ (np.abs(Bmid) + Brad)
        term3 = np.abs(Amid) @ Brad
        Crad = term1 + term2 + term3

        # 4. Final Bounds
        setround(-1)
        C_inf = Cmid - Crad
        
        setround(1)
        C_sup = Cmid + Crad
        
        setround(0)
        return Interval(C_inf, C_sup)
    
    def __truediv__(self, other):
            if not isinstance(other, Interval):
                other = Interval(other)

            # 2. Check for 0 in denominator (s standard division requires 0 not in B)
            if np.any((other.inf <= 0) & (other.sup >= 0)):
                raise ZeroDivisionError("Interval division by zero not yet supported")

            # 3. Compute 1/B = [1/b_sup, 1/b_inf]
            setround(-1) # Round Down
            recip_inf = 1.0 / other.sup  # 1/b2 (Alefeld's formula)
            
            setround(1)  # Round Up
            recip_sup = 1.0 / other.inf  # 1/b1 (Alefeld's formula)
            
            setround(0)

            # 4. Compute A * (1/B)
            reciprocal = Interval(recip_inf, recip_sup)
            return self * reciprocal
    
    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        exponent = int(exponent)
        
        if exponent == 0:
             return Interval(np.ones_like(self.inf), np.ones_like(self.sup))
             
        if exponent % 2 == 0:
            # Even power
            abs_inf = np.abs(self.inf)
            abs_sup = np.abs(self.sup)
            max_abs = np.maximum(abs_inf, abs_sup)
            
            # Check for 0 containment
            has_zero = (self.inf <= 0) & (self.sup >= 0)
            min_abs = np.minimum(abs_inf, abs_sup)
            
            setround(-1)
            lo = min_abs ** exponent
            if np.ndim(lo) == 0:
                if has_zero: lo = 0.0
            else:
                lo[has_zero] = 0.0
                
            setround(1)
            hi = max_abs ** exponent
            setround(0)
            return Interval(lo, hi)
        else:
            # Odd power
            setround(-1); lo = self.inf ** exponent
            setround(1); hi = self.sup ** exponent
            setround(0)
            return Interval(lo, hi)
    
    # --- Indexing and Slicing Support ---

    def __getitem__(self, index):
        """
        Allows accessing elements like x[0] or slicing like x[1:5].
        Returns a new Interval composed of the selected elements.
        """
        return Interval(self.inf[index], self.sup[index])

    def __setitem__(self, index, value):
        """
        Allows assigning elements like x[0] = Interval(1, 2).
        """
        if not isinstance(value, Interval):
            value = Interval(value)
        
        # We assign the bounds directly to the underlying numpy arrays
        self.inf[index] = value.inf
        self.sup[index] = value.sup

    def __iter__(self):
        """
        Allows iterating over the interval vector: 'for x_i in x:'
        """
        for i in range(len(self)):
            yield self[i]

    # --- Utility Methods ---

    def hull(self, other):
        """Convex hull of two intervals."""
        if not isinstance(other, Interval):
            other = Interval(other)
        return Interval(np.minimum(self.inf, other.inf), np.maximum(self.sup, other.sup))

    def in0(self, other):
        """
        Checks if 'self' is strictly in the interior of 'other'.
        Returns True only if self is entirely inside other.
        Used for convergence checks.
        """
        return np.all(other.inf < self.inf) and np.all(self.sup < other.sup)
    
    # --- Add these to Interval class in intlab.py ---

    def intersect(self, other):
        """
        Returns NaN interval if empty. WARNING: 
        NaN can be continued from here to other calculations, propagating errors.
        If a NaN value is expected at the end, proceed, otherwise be careful using this function.
        """
        if not isinstance(other, Interval):
            other = Interval(other)
        
        new_inf = np.maximum(self.inf, other.inf)
        new_sup = np.minimum(self.sup, other.sup)
        
        # Check for empty intersection
        mask = new_inf > new_sup
        if np.any(mask):
            new_inf[mask] = np.nan
            new_sup[mask] = np.nan
            
        return Interval(new_inf, new_sup)

    @property
    def mag(self):
        return np.maximum(np.abs(self.inf), np.abs(self.sup))

    @property
    def mig(self):
        """Mignitude: min(abs(x)) for x in interval"""
        # If 0 is inside, mignitude is 0.
        is_inside = (self.inf <= 0) & (self.sup >= 0)
        result = np.minimum(np.abs(self.inf), np.abs(self.sup))
        result[is_inside] = 0.0
        return result

    def inverse(self):
        """
        Computes enclosure of A^-1 by solving A * X = I
        """
        from solvers import verifylss
        n = self.shape[0]
        I = Interval(np.eye(n))
        return verifylss(self, I)

    # --- Standard Functions (Monotonic Examples) ---
    
    def exp(self):
        setround(-1)
        lo = np.exp(self.inf)
        setround(1)
        hi = np.exp(self.sup)
        setround(0)
        return Interval(lo, hi)

    def log(self):
        if np.any(self.inf <= 0):
             raise ValueError("Log undefined for non-positive values")
        setround(-1)
        lo = np.log(self.inf)
        setround(1)
        hi = np.log(self.sup)
        setround(0)
        return Interval(lo, hi)
        
    def sqrt(self):
        """Rigorous Square Root"""
        if np.any(self.inf < 0):
            raise ValueError("Sqrt of negative interval")
        
        setround(-1)
        lo = np.sqrt(self.inf)
        setround(1)
        hi = np.sqrt(self.sup)
        setround(0)
        return Interval(lo, hi)
        
    def sin(self):
        """
        Rigorous Interval Sine.
        
        Handles non-monotonicity by explicitly checking for:
        1. Peaks (pi/2 + 2k*pi) -> Sets upper bound to 1.0
        2. Troughs (3pi/2 + 2k*pi) -> Sets lower bound to -1.0
        """
        # Constants
        pi = np.pi
        two_pi = 2 * pi
        half_pi = 0.5 * pi
        
        # 1. Evaluate Sine at the endpoints with directed rounding
        # Note: We compute both bounds for both endpoints to ensure inclusion 
        # even if the hardware sin implementation isn't perfectly rigorous.
        setround(-1) # Down
        lo1 = np.sin(self.inf)
        lo2 = np.sin(self.sup)
        
        setround(1)  # Up
        hi1 = np.sin(self.inf)
        hi2 = np.sin(self.sup)
        setround(0)
        
        # The initial candidate bounds are the min/max of the endpoint values
        curr_lo = np.minimum(lo1, lo2)
        curr_hi = np.maximum(hi1, hi2)

        # 2. Check for "Peaks" (Max = 1.0)
        # A peak occurs at: x = 2*k*pi + pi/2
        # We assume a peak is inside if: ceil((inf - pi/2)/2pi) * 2pi + pi/2 <= sup
        k_max = np.ceil((self.inf - half_pi) / two_pi)
        peak_loc = k_max * two_pi + half_pi
        
        # If the calculated peak location is <= sup, then the interval contains a peak
        has_peak = peak_loc <= self.sup
        
        # 3. Check for "Troughs" (Min = -1.0)
        # A trough occurs at: x = 2*k*pi + 3pi/2
        k_min = np.ceil((self.inf - 3*half_pi) / two_pi)
        trough_loc = k_min * two_pi + 3*half_pi
        
        has_trough = trough_loc <= self.sup

        # 4. Handle "Wide" Intervals (Width >= 2*pi)
        # If the interval is wider than a full period, it definitely contains both.
        width = self.sup - self.inf
        is_full_cycle = width >= two_pi
        
        # If we have a peak, the upper bound is exactly 1.0
        # (We use np.where to handle both scalar and vector inputs safely)
        curr_hi = np.where(has_peak | is_full_cycle, 1.0, curr_hi)
        
        # If we have a trough, the lower bound is exactly -1.0
        curr_lo = np.where(has_trough | is_full_cycle, -1.0, curr_lo)

        return Interval(curr_lo, curr_hi)
            
    def cos(self):
        """
        Rigorous Interval Cosine.
        Implemented as sin(x + pi/2).
        """
        # cos(x) = sin(x + pi/2)
        shifted = self + (np.pi * 0.5)
        return shifted.sin()

# --- Global Helper ---
def midrad(mid, rad):
    """Constructor for midpoint-radius."""
    if isinstance(mid, Interval):
        # If mid is passed as Interval, extract its midpoint values
        mid = mid.mid
        
    rad = np.abs(rad)
    return Interval(mid - rad, mid + rad)


def get_power_10_interval(exponent):
    """
    Returns a rigorous Interval for 10^exponent.
    """
    if exponent in _POWER_10_CACHE:
        return _POWER_10_CACHE[exponent]

    # We compute 10^exponent using Python's float pow with directed rounding.
    # Python's float arithmetic is IEEE-754 compliant, so pow() works 
    # if we wrap it in rounding modes.
    
    setround(-1)
    lo = pow(10.0, exponent)
    
    setround(1)
    hi = pow(10.0, exponent)
    
    setround(0)
    
    # Store and return
    result = Interval(lo, hi)
    _POWER_10_CACHE[exponent] = result
    return result

def parse_decimal_string(s):
    """
    Parses a float string into integer mantissa and exponent.
    Example: "-1.23e-4" -> mantissa -123, exponent -6
    Logic: -1.23e-4 = -123 * 10^-2 * 10^-4 = -123 * 10^-6
    """
    s = s.lower().replace('_', '') 
    if 'e' in s:
        base, exppart = s.split('e')
        exponent = int(exppart)
    else:
        base = s
        exponent = 0
        
    if '.' in base:
        # Count decimal places
        integer_part, fractional_part = base.split('.')
        # Exponent decreases by the length of fractional part
        exponent -= len(fractional_part)
        # Reassemble mantissa as integer
        mantissa_str = integer_part + fractional_part
    else:
        mantissa_str = base
        
    return int(mantissa_str), exponent

def str2interval(s):
    """
    Rigorous string-to-interval conversion using integer arithmetic 
    and verified powers of 10.
    Implementation of Algorithm 4.2 / 4.3 from INTLAB - INTERVAL LABORATORY by Rump.
    """
    try:
        # 1. Decompose string into integers m * 10^e
        mantissa, exponent = parse_decimal_string(s)
        
        # 2. Get interval for the integer mantissa
        # Integers are exact in Python, but we need to convert to float 
        # bounds carefully if they are huge (though Python floats handle up to 10^308).
        # For simple cases, Interval(float(m)) is actually safe IF m is within 
        # 53 bits of precision. If m is huge, we need directed rounding for it too.
        # Currently this direct rounding for huge m is not implemented.
        
        setround(-1)
        m_lo = float(mantissa)
        setround(1)
        m_hi = float(mantissa)
        setround(0)
        
        m_interval = Interval(m_lo, m_hi)
        
        # 3. Get interval for 10^exponent
        pow10_interval = get_power_10_interval(exponent)
        
        # 4. Multiply: result = mantissa * 10^exponent
        return m_interval * pow10_interval

    except Exception as e:
        # Fallback or error handling
        setround(0)
        raise ValueError(f"parsing failed for '{s}': {e}")