# COREQ
An implementation of the COREQ (CORrelation EQuivalence) algorithm to compute low redundancy estimates for large correlation matrices. The algorithm is described and analysed in detail in the paper:

Erik Scharwächter, Fabian Geier, Lukas Faber, Emmanuel Müller. **Low Redundancy Estimation of Correlation Matrices for Time Series using Triangular Bounds.** In: *Proceedings of the 22nd Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)*, 2018.

## Contact and citation
* Corresponding author: Erik Scharwächter <erik.scharwaechter@hpi.de>
* Please cite our paper if you use or modify our code for your own work.

## Description
Please check our our project website at https://hpi.de/mueller/coreq.html for more information on the method, supplementary material and visualizations.

## Installation
Automatically build and install the BlockCorr module for Python2 and Python3 with
```
$ make
$ make install
```

or use the setup.py script (check --help for details).

## Usage
Example: A dataset of 1000 time series (length 50) that consists of two perfectly separated clusters of size 700 and 300.
```
>>> import numpy as np
>>> import BlockCorr
>>> np.random.seed(0)
>>> X = np.random.permutation(np.concatenate((np.repeat(np.random.randn(1, 50), 700, axis=0),
            np.repeat(np.random.randn(1, 50), 300, axis=0))))
>>> X.shape
(1000, 50)
>>> alpha = 0.8
>>> labels, pivots, pivot_corr_triu, computations = BlockCorr.COREQ(X, BlockCorr.ESTIMATE_AVERAGE, alpha)
clustering finished with 1300 correlation computations --- 2 clusters detected
>>> pivot_corr_triu
array([ 1.        , -0.05918059,  1.        ])

>>> # full pivot correlation matrix:
>>> pivot_corr = np.zeros((len(pivots), len(pivots)))
>>> pivot_corr[np.triu_indices(len(pivots))] = pivot_corr_triu
>>> pivot_corr[np.tril_indices(len(pivots))] = pivot_corr_triu

>>> # full time series correlation matrix:
>>> cluster_matrix = np.zeros((X.shape[0], len(pivots)))
>>> cluster_matrix[range(0,len(labels)), labels] = 1
>>> R = cluster_matrix.dot(pivot_corr).dot(cluster_matrix.transpose())
>>> np.abs((cluster_matrix.dot(pivot_corr).dot(cluster_matrix.transpose()) - np.corrcoef(X))).sum()
0.0
```

## License
The code is released under the MIT license.
