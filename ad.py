import numpy as np
from pyia import Interval

class Gradient:
    def __init__(self, value, deriv=None):
        """
        Represents a value and its derivative: f(x), f'(x).
        value: Scalar, Array, or Interval
        deriv: The derivative (Jacobian). Defaults to 0 or Identity.
        """
        self.x = value
        
        # If no derivative provided, assume constant (deriv=0)
        if deriv is None:
            # Handle shapes for scalar vs array
            shape = np.shape(value)
            if shape == ():
                self.dx = 0.0
            else:
                self.dx = np.zeros_like(value)
        else:
            self.dx = deriv

    def __repr__(self):
        return f"Gradient(value={self.x}, dx={self.dx})"

    # --- Arithmetic Operator Overloading (The Chain Rule) ---

    def __add__(self, other):
        if not isinstance(other, Gradient):
            # g(x) + c -> (g + c, g')
            return Gradient(self.x + other, self.dx)
        # u + v -> (u+v, u' + v')
        return Gradient(self.x + other.x, self.dx + other.dx)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Gradient):
            return Gradient(self.x - other, self.dx)
        return Gradient(self.x - other.x, self.dx - other.dx)

    def __rsub__(self, other):
        # c - g(x) -> (c - g, -g')
        return Gradient(other - self.x, -self.dx)

    def __mul__(self, other):
        if not isinstance(other, Gradient):
            # c * g(x) -> (c*g, c*g')
            return Gradient(self.x * other, self.dx * other)
        
        # Product Rule
        new_val = self.x * other.x
        new_dx = (self.dx * other.x) + (self.x * other.dx)
        return Gradient(new_val, new_dx)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Gradient):
            # g(x) / c -> (g/c, g'/c)
            return Gradient(self.x / other, self.dx / other)
        
        # Quotient Rule 
        new_val = self.x / other.x
        num = (self.dx * other.x) - (self.x * other.dx)
        den = other.x * other.x
        return Gradient(new_val, num / den)

    def __pow__(self, power):
        """
        Support for x**n (Power Rule).
        d/dx (u^n) = n * u^(n-1) * u'
        """
        if isinstance(power, (int, float)):
            val = self.x ** power
            # Derivative: n * u^(n-1) * dx
            deriv = power * (self.x ** (power - 1)) * self.dx
            return Gradient(val, deriv)
        raise NotImplementedError("Gradient power only supports scalar exponents")
    
    def __getitem__(self, index):
        """
        Extracts the i-th component of the vector function.
        Value is scalar x[i].
        Derivative is row dx[i, :].
        """
        return Gradient(self.x[index], self.dx[index])

    # --- Elementary Functions (Chain Rule) ---

    def exp(self):
        # d/dx exp(u) = exp(u) * u'
        ex = np.exp(self.x) if not isinstance(self.x, Interval) else self.x.exp()
        return Gradient(ex, ex * self.dx)
    
    def log(self):
        # d/dx log(u) = (1/u) * u'
        lg = np.log(self.x) if not isinstance(self.x, Interval) else self.x.log()
        return Gradient(lg, self.dx / self.x)

    def sin(self):
        # d/dx sin(u) = cos(u) * u'
        s = np.sin(self.x) if not isinstance(self.x, Interval) else self.x.sin()
        c = np.cos(self.x) if not isinstance(self.x, Interval) else self.x.cos()
        return Gradient(s, c * self.dx)

    def cos(self):
        # d/dx cos(u) = -sin(u) * u'
        c = np.cos(self.x) if not isinstance(self.x, Interval) else self.x.cos()
        s = np.sin(self.x) if not isinstance(self.x, Interval) else self.x.sin()
        return Gradient(c, -s * self.dx)
    
    def sqrt(self):
        # d/dx sqrt(u) = (1 / 2*sqrt(u)) * u'
        sq = np.sqrt(self.x) if not isinstance(self.x, Interval) else self.x.sqrt()
        return Gradient(sq, self.dx / (2 * sq))
    

def initvar(x):
    """
    Initialises independent variables for AD.
    If x is a vector of length N, returns a Gradient vector
    where the Jacobian is the Identity matrix (NxN).
    """
    x = np.array(x) if not isinstance(x, Interval) else x
    n = len(x)

    # Jacobian is Identity: dx[i]/dx[j] = 1 if i==j else 0
    return Gradient(x, np.eye(n))