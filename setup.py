"""Build the optional Cython kernel.

Everything else about the distribution -- name, version, dependencies, console
script -- lives in ``pyproject.toml``.  This file exists only for
``parvaneh._cython_backend``: it has to be cythonised, which the declarative
configuration in ``pyproject.toml`` cannot express.

The extension is optional.  If no C compiler (or no Cython) is available the
build still succeeds, and ``parvaneh.available_backends()`` simply reports
``"cython": False``.
"""

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                "parvaneh._cython_backend",
                ["src/parvaneh/_cython_backend.pyx"],
                include_dirs=[numpy.get_include()],
                optional=True,
            )
        ],
        language_level=3,
    )
)
