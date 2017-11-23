import numpy as np, pandas as pd
import BlockCorr
from netCDF4 import Dataset

datafile = "StarLightCurves_TRAIN"
dataset = pd.read_csv(datafile, sep=",", header=None)
dataset.drop(0, axis=1, inplace=True)

np.random.seed(0)

alpha = 0.1
print("alpha = %.2f" % alpha)
labels, pivots, pivot_corr_triu, computations = BlockCorr.COREQ(dataset, BlockCorr.ESTIMATE_AVERAGE, alpha)
l1, l2, lmax, elems = BlockCorr.Loss(dataset, pivot_corr_triu, labels)
print("--- output  %.5f" % (l1/elems))
print("+++ must be 0.25437")

alpha = 0.5
print("alpha = %.2f" % alpha)
labels, pivots, pivot_corr_triu, computations = BlockCorr.COREQ(dataset, BlockCorr.ESTIMATE_AVERAGE, alpha)
l1, l2, lmax, elems = BlockCorr.Loss(dataset, pivot_corr_triu, labels)
print("--- output  %.5f" % (l1/elems))
print("+++ must be 0.18985")

alpha = 0.9
print("alpha = %.2f" % alpha)
labels, pivots, pivot_corr_triu, computations = BlockCorr.COREQ(dataset, BlockCorr.ESTIMATE_AVERAGE, alpha)
l1, l2, lmax, elems = BlockCorr.Loss(dataset, pivot_corr_triu, labels)
print("--- output  %.5f" % (l1/elems))
print("+++ must be 0.09261")

