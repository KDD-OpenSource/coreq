from distutils.core import setup, Extension
import numpy

mod = Extension('BlockCorr',
    include_dirs = [numpy.get_include()],
    sources = ['BlockCorr.c', 'list.c'],
    extra_compile_args=['-fopenmp','-O3','-march=native','-mavx','-funroll-loops'],
    #extra_compile_args=['-fopenmp','-g','-pg'],
    extra_link_args=['-lgomp']
)

setup (name = 'BlockCorr',
    author = 'Erik Scharwaechter',
    author_email = 'erik.scharwaechter@hpi.de',
    ext_modules = [mod]
)
