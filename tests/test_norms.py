import BlockCorr
import numpy as np

N = 100
D = 1000

np.random.seed(0)

### WITHOUT CLUSTERS ###

foo = np.random.rand(N, D)
foo_corr = BlockCorr.PearsonTriu(foo)

membs = range(N)
dists = BlockCorr.Loss(foo_corr, foo_corr, membs, True)
print "Identity clustering"
print "--- output ", dists
print "+++ must be (0.0, 0.0, 0.0, 5050)"

membs = np.random.permutation(N)
max_clus = np.max(membs)
representatives = [foo[np.random.choice(np.where(membs == k)[0])] for k in range(0, max_clus+1)]
cluster_corr = BlockCorr.PearsonTriu(representatives)
dists = BlockCorr.Loss(foo_corr, cluster_corr, membs, True)
print "Random permutation clustering"
print "--- output ", dists
print "+++ must be (0.0, 0.0, 0.0, 5050)"

print "COREQ clustering (alpha=0.1)"
membs, pivots, pivot_corr_triu, comps = BlockCorr.COREQ(foo, BlockCorr.ESTIMATE_PIVOT, 0.1)
dists = BlockCorr.Loss(foo_corr, pivot_corr_triu, membs, True)
print "--- output ", dists
print "+++ must be (8.36966696626563, 1.949955178414159, 0.8965384221246233, 5050)"

### WITH CLUSTERS ###

bar = np.concatenate([
        np.random.rand(1, D).repeat(N/2, axis=0),
        np.random.rand(1, D).repeat(N/2, axis=0),
        np.random.rand(20, D),
      ])
bar_corr = BlockCorr.PearsonTriu(bar)

print "COREQ clustering with group structures (alpha=0.1)"
membs, pivots, pivot_corr_triu, comps = BlockCorr.COREQ(bar, BlockCorr.ESTIMATE_PIVOT, 0.1)

dists = BlockCorr.Loss(bar_corr, pivot_corr_triu, membs, True)
print "--- output ", dists
print "+++ must be (0.0, 0.0, 0.0, 7260)"

dists = BlockCorr.Loss(bar, pivot_corr_triu, membs, False)
print "--- output ", dists
print "+++ must be (0.0, 0.0, 0.0, 7260)"
