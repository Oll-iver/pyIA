# pyIA
# PyIA: Interval Arithmetic Library

**PyIA** is a Python library for **interval arithmetic**. Unlike standard floating-point arithmetic, which is prone to rounding errors, PyIA uses **Interval Arithmetic** with **IEEE 754 directed rounding** to produce mathematically guaranteed bounds for every computation.

If PyIA returns an interval `[a, b]`, the true mathematical result is **guaranteed** to be inside it.

## 🚀 Key Features

* **Native Directed Rounding:** Custom C-extension (`rounding.c`) to control the FPU for rigorous lower/upper bound calculation.
* **Interval Arithmetic:** Full operator overloading (`+`, `-`, `*`, `/`, `**`, `@`) for scalars, vectors, and matrices.
* **Sparse Support:** Memory-efficient `SparseIntervalMatrix` class supporting operations on systems with 50,000+ variables.
* **Automatic Differentiation (AD):** Vectorised forward-mode AD (`Gradient` class) capable of handling mixed Interval/Float operations.
* **Verified Solvers:**
    * **Linear Systems ($Ax=b$):** Uses Rump's algorithm (Hansen-Bliek-Rump) to bound solutions to ill-conditioned systems (e.g., Hilbert matrices).
    * **Nonlinear Systems ($F(x)=0$):** Implements the **Newton-Krawczyk** operator to prove the existence and uniqueness of roots.

## 🛠️ Installation

PyIA requires a C compiler (GCC/Clang) to build the rounding control module.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/PyIA.git](https://github.com/YOUR_USERNAME/PyIA.git)
    cd PyIA
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install numpy scipy pytest
    ```

3.  **Compile the Rounding Kernel:**
    The Python logic relies on this C-extension to switch rounding modes (Up/Down/Nearest).
    ```bash
    python build_rounding.py
    ```
    *Success Check:* You should see a `.so` file generated in the directory (e.g., `rounding.cpython-312-darwin.so`).

## ⚡ Technical Report and Quick Start
Please view the attached PDF 'pyIA Technical Report.pdf' for a technical report, detailing the algorithms used, performance benchmarks, solvers of systems such as **A**x=**b** and a **quickstart guide.**
