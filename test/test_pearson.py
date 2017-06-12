from time import clock, time
import numpy as np
import coreq

n = 1000
r = np.random.random((n, n))

u, s = clock(), time()
a = np.corrcoef(r)
v, t = clock(), time()

print('numpy.corrcoef():')
print('process time: %.1f sec.' % (v - u))
print('wall time: %.1f sec.' % (t - s))

b = a[np.triu_indices(n, 1)]
c = a[np.triu_indices(n, 0)]

del a
u, s = clock(), time()
a = coreq.Pearson(r)
v, t = clock(), time()

print('------------------------------------')
print('coreq.Pearson():')
print('process time: %.1f sec.' % (v - u))
print('wall time: %.1f sec.' % (t - s))
print('------------------------------------')
print('Maximum deviation: %e' % (np.max(abs(b - a[np.triu_indices(n, 1)]))))

del a
u, s = clock(), time()
a = coreq.PearsonTriu(r, False, False)
v, t = clock(), time()

print('------------------------------------')
print('coreq.PearsonTriu(diagonal=False, mmap=False):')
print('process time: %.1f sec.' % (v - u))
print('wall time: %.1f sec.' % (t - s))
print('------------------------------------')
print('Maximum deviation: %e' % (np.max(abs(b - a))))

del a
u, s = clock(), time()
a = coreq.PearsonTriu(r, False, True)
v, t = clock(), time()

print('------------------------------------')
print('coreq.PearsonTriu(diagonal=False, mmap=True):')
print('process time: %.1f sec.' % (v - u))
print('wall time: %.1f sec.' % (t - s))
print('------------------------------------')
print('Maximum deviation: %e' % (np.max(abs(b - a))))

del a
u, s = clock(), time()
a = coreq.PearsonTriu(r, True, False)
v, t = clock(), time()

print('------------------------------------')
print('coreq.PearsonTriu(diagonal=True, mmap=False):')
print('process time: %.1f sec.' % (v - u))
print('wall time: %.1f sec.' % (t - s))
print('------------------------------------')
print('Maximum deviation: %e' % (np.max(abs(c - a))))

