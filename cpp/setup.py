
"""
setup.py
========
Build the qr_engine C++ extension (.pyd on Windows, .so on Linux/Mac).

Usage:
    cd cpp
    pip install .
    # or
    python setup.py build_ext --inplace
"""

import platform
import pybind11
from setuptools import setup, Extension

ext = Extension(
    "qr_engine",
    sources=["bindings.cpp", "ssvi.cpp", "greeks.cpp"],
    include_dirs=[".", pybind11.get_include()],
    language="c++",
)

if platform.system() == "Windows":
    ext.extra_compile_args = ["/Od", "/Zi", "/std:c++17", "/EHsc"]
    ext.extra_link_args = ["/DEBUG"]
else:
    ext.extra_compile_args = ["-O3", "-std=c++17", "-fvisibility=hidden"]

setup(
    name="qr_engine",
    version="1.0.0",
    description="QR Equity Derivatives Flow - C++ Engine",
    ext_modules=[ext],
)
