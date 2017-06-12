from distutils.core import setup, Extension
import numpy

mod = Extension('coreq',
    include_dirs = [numpy.get_include()],
    sources = ['coreq.c', 'list.c'],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp']
)

setup (name = 'coreq',
    author = 'Erik Scharwaechter',
    author_email = 'erik.scharwaechter@hpi.de',
    description = 'COREQ: Low redundancy estimation of correlation matrices',
    version = '0.3',
    ext_modules = [mod]
)
