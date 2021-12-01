from setuptools import setup
from setuptools import Extension

import numpy

setup(
    name="example",
    version="1.0",
    include_dirs=[numpy.get_include()],
    ext_modules=[Extension("example", ["example.c"])]
)