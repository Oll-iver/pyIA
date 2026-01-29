# pyia_kernel.py 

try:
    import rounding
except ImportError:
    raise ImportError("Compile the rounding module first using build_rounding.py!")

def setround(mode):
    # The C-extension handles the logic safely
    rounding.setround(mode)

def getround():
    return rounding.getround()

class RoundingMode:
    """
    Context manager for safe rounding mode changes.
    Ensures mode is restored even if exceptions occur.
    """
    def __init__(self, mode):
        self.target_mode = mode
        self.prev_mode = 0  # Default to nearest

    def __enter__(self):
        self.prev_mode = rounding.getround() 
        rounding.setround(self.target_mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        rounding.setround(self.prev_mode)

