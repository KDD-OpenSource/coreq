import coreq
import numpy as np
import sys

N = 100
D = 1000
mmap_arr = True

foo = np.random.rand(N, D)
foo_corr = coreq.PearsonTriu(foo, True)

membs = range(N)
dists = coreq.Norms(foo_corr, foo_corr, membs, True)
print "Identity clustering"
print membs
print np.unique(membs)
print dists
print "valid correlations (not NaN): %d" % np.where(~np.isnan(foo_corr))[0].size

membs = np.random.permutation(N)
max_clus = np.max(membs)
representatives = [foo[np.random.choice(np.where(membs == k)[0])] for k in range(0, max_clus+1)]
cluster_corr = coreq.PearsonTriu(representatives, True, mmap_arr)
dists = coreq.Norms(foo_corr, cluster_corr, membs, True)
print "Random permutation clustering"
print membs
print np.unique(membs)
print dists
print "valid correlations (not NaN): %d" % np.where(~np.isnan(foo_corr))[0].size

membs = coreq.Cluster(foo, 0.71, 1, 0)
max_clus = np.max(membs)
representatives = [
        foo[np.random.choice(np.where(membs == k)[0])]
        if (np.where(membs == k)[0].size > 0)
        else ([np.nan]*D)
    for k in range(0, max_clus+1)]
cluster_corr = coreq.PearsonTriu(representatives, True, mmap_arr)
cluster_corr[0:(max_clus+1)] = 0.0 # noise cluster always has 0.0 correlation
dists = coreq.Norms(foo_corr, cluster_corr, membs, True)
print "Transitivity clustering (kappa=1)"
print membs
print np.unique(membs)
print dists
print "valid correlations (not NaN): %d" % np.where(~np.isnan(foo_corr))[0].size

membs = coreq.Cluster(foo, 0.71, 2, 0)
max_clus = np.max(membs)
representatives = [
        foo[np.random.choice(np.where(membs == k)[0])]
        if (np.where(membs == k)[0].size > 0)
        else ([np.nan]*D)
    for k in range(0, max_clus+1)]
cluster_corr = coreq.PearsonTriu(representatives, True, mmap_arr)
cluster_corr[0:(max_clus+1)] = 0.0 # noise cluster always has 0.0 correlation
dists = coreq.Norms(foo_corr, cluster_corr, membs, True)
print "Transitivity clustering (kappa=2)"
print membs
print np.unique(membs)
print dists
print "valid correlations (not NaN): %d" % np.where(~np.isnan(foo_corr))[0].size


bar = np.concatenate([
        np.random.rand(1, D).repeat(N/2, axis=0),
        np.random.rand(1, D).repeat(N/2, axis=0),
        np.random.rand(20, D),
      ])
bar_corr = coreq.PearsonTriu(bar, True, mmap_arr)
membs = coreq.Cluster(bar, 0.71, 2, 0)
max_clus = np.max(membs)
representatives = [
        bar[np.random.choice(np.where(membs == k)[0])]
        if (np.where(membs == k)[0].size > 0)
        else ([np.nan]*D)
    for k in range(0, max_clus+1)]
cluster_corr = coreq.PearsonTriu(representatives, True, mmap_arr)
cluster_corr[0:(max_clus+1)] = 0.0 # noise cluster always has 0.0 correlation
dists = coreq.Norms(bar_corr, cluster_corr, membs, True)
print "Transitivity clustering with actual clusters (kappa=2)"
print membs
print np.unique(membs)
print dists
print "valid correlations (not NaN): %d" % np.where(~np.isnan(bar_corr))[0].size

dists = coreq.Norms(bar, cluster_corr, membs, False)
print "Transitivity clustering with actual clusters (kappa=2, on-the-fly correlations)"
print membs
print np.unique(membs)
print dists
print "valid correlations (not NaN): %d" % np.where(~np.isnan(bar_corr))[0].size
