from setuptools import setup
from setuptools import Extension

import numpy

setup(
    name="sparsedot",
    version="0.0.1",
    include_dirs=[numpy.get_include()],
    ext_modules=[Extension("sparsedot", ["sparsedot.c"])]
)