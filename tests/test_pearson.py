from time import clock, time
import numpy as np
import BlockCorr

n = 5000
r = np.random.random((n, n))

u, s = clock(), time()
a = np.corrcoef(r)
v, t = clock(), time()
print('numpy.corrcoef():')
print('process time: %.1f sec.' % (v - u))
print('wall time: %.1f sec.' % (t - s))
c = a[np.triu_indices(n, 0)]

del a
u, s = clock(), time()
a = BlockCorr.Pearson(r)
v, t = clock(), time()
print('------------------------------------')
print('BlockCorr.Pearson():')
print('process time: %.1f sec.' % (v - u))
print('wall time: %.1f sec.' % (t - s))
print('------------------------------------')
print('--- Maximum deviation: %e' % (np.max(abs(c - a[np.triu_indices(n, 0)]))))
print('+++ Should be 0 (up to numerical precision)')

del a
u, s = clock(), time()
a = BlockCorr.PearsonTriu(r)
v, t = clock(), time()
print('------------------------------------')
print('BlockCorr.PearsonTriu():')
print('process time: %.1f sec.' % (v - u))
print('wall time: %.1f sec.' % (t - s))
print('------------------------------------')
print('--- Maximum deviation: %e' % (np.max(abs(c - a))))
print('+++ Should be 0 (up to numerical precision)')

