from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize
import numpy

setup(name="accelerated-phase-unwrapping", version="0.1.0a1", packages=find_packages(),
      ext_modules=cythonize([Extension("accelerated_unwrap._cython_backend",
                                      ["accelerated_unwrap/_cython_backend.pyx"],
                                      include_dirs=[numpy.get_include()],
                                      optional=True)], language_level=3))
