How to build and install
========================

Simple wrapper to build and install the BlockCorr module for Python2 and Python3:
1) make
2) make install

... with Intel C++ compiler
1) In setup.py, replace both -fopenmp with -qopenmp
2) CC=icc LDSHARED="icc -shared" make
3) make install
4) LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/icc/libs/ python ...

