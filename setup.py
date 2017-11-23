from distutils.core import setup, Extension
import numpy, os

def read(fname):
  return open(os.path.join(os.path.dirname(__file__), fname)).read()

mod = Extension('BlockCorr',
  include_dirs = [numpy.get_include()],
  sources = ['BlockCorr.c', 'list.c', 'PyBlockCorr.c'],
  extra_compile_args=['-fopenmp','-O3','-march=native','-mavx','-funroll-loops'],
  #extra_compile_args=['-fopenmp','-O3'],
  #extra_compile_args=['-fopenmp','-g','-pg'],
  extra_link_args=['-lgomp']
)

setup(
  name = 'BlockCorr',
  version = '0.3',
  author = 'Erik Scharwaechter',
  author_email = 'erik.scharwaechter@hpi.de',
  license = 'MIT',
  long_description=read('README'),
  ext_modules = [mod]
)
