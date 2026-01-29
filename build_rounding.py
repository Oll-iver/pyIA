# CURRENTLY THIS WORKS ONLY ON MACOS AND LINUX
import os
import sys
from setuptools import setup, Extension

# 1. Define the C source code for the extension
c_source = """
#include <Python.h>
#include <fenv.h>

// Macro to check for macOS/Linux constants
#ifndef FE_UPWARD
  #define FE_UPWARD 0x0800
#endif
#ifndef FE_DOWNWARD
  #define FE_DOWNWARD 0x0400
#endif
#ifndef FE_TONEAREST
  #define FE_TONEAREST 0x0000
#endif

static PyObject* set_round(PyObject* self, PyObject* args) {
    int mode;
    if (!PyArg_ParseTuple(args, "i", &mode))
        return NULL;

    int result = -1;
    if (mode == 1) {
        result = fesetround(FE_UPWARD);
    } else if (mode == -1) {
        result = fesetround(FE_DOWNWARD);
    } else if (mode == 0) {
        result = fesetround(FE_TONEAREST);
    }

    if (result != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set rounding mode via fenv.h");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* get_round(PyObject* self, PyObject* args) {
    int mode = fegetround();
    return Py_BuildValue("i", mode);
}

// Module definition
static PyMethodDef RoundingMethods[] = {
    {"setround", set_round, METH_VARARGS, "Set rounding mode: 1=Up, -1=Down, 0=Nearest"},
    {"getround", get_round, METH_VARARGS, "Get current rounding mode constant"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef roundingmodule = {
    PyModuleDef_HEAD_INIT,
    "rounding",
    "Native rounding control for macOS",
    -1,
    RoundingMethods
};

PyMODINIT_FUNC PyInit_rounding(void) {
    return PyModule_Create(&roundingmodule);
}
"""

# 2. Write source to file
with open("rounding.c", "w") as f:
    f.write(c_source)

# 3. Create setup.py structure manually and run build
# We use a subprocess to ensure it runs in the correct environment context
import subprocess

setup_code = """
from setuptools import setup, Extension
module = Extension('rounding', sources=['rounding.c'])
setup(
    name='rounding',
    version='1.0',
    description='Native FPU rounding control',
    ext_modules=[module]
)
"""

with open("setup_builder.py", "w") as f:
    f.write(setup_code)

print("Compiling Native Rounding Extension...")
try:
    subprocess.check_call([sys.executable, 'setup_builder.py', 'build_ext', '--inplace'])
    print("Compilation Successful!")
    print("Move the generated .so file (inside build/...) to your working folder or rely on the local import.")
except subprocess.CalledProcessError:
    print("Compilation Failed. Ensure you have Xcode command line tools installed (xcode-select --install).")