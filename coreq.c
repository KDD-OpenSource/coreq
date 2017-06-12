/* vim:set ts=8 sw=2 sts=2 noet:  */

/*
Copyright (C) 2017 Erik Scharwaechter <erik.scharwaechter@hpi.de>
-- based on the code of CorrCoef (https://github.com/UP-RS-ESP/CorrCoef)
-- Copyright (C) 2016 Rheinwalt

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "list.h"

#define VERSION "0.3"

// compute pearson correlation coefficient between time series at positions i1 and i2 in d (of length l)
// NOTE: result may be nan, if the variance of any of the time series is zero, or if
// any of the time series contains nans
double pearson2(const double *d, const unsigned long i1, const unsigned long i2, const unsigned long l) {
  unsigned int i;
  double mean1, mean2, var1, var2, cov;

  // compute means
  mean1 = 0.0; mean2 = 0.0;
  for (i = 0; i < l; i++) {
    mean1 += d[i1*l+i]; mean2 += d[i2*l+i];
  }
  mean1 /= l; mean2 /= l;

  // compute variances and covariance
  var1 = 0.0; var2 = 0.0;
  cov = 0.0;
  for (i = 0; i < l; i++) {
    var1 += (d[i1*l+i]-mean1)*(d[i1*l+i]-mean1);
    var2 += (d[i2*l+i]-mean2)*(d[i2*l+i]-mean2);
    cov += (d[i1*l+i]-mean1)*(d[i2*l+i]-mean2);
  }
  var1 /= (l-1); var2 /= (l-1); cov /= (l-1); // denominators don't really matter

  // compute correlation
  return cov/(sqrt(var1)*sqrt(var2));
}

// compute n-by-n correlation matrix for complete data set d with n rows and l columns
PyArrayObject *
pearson(const double *d, unsigned long n, unsigned long l) {
  PyArrayObject *coef;
  long int dim[2];
  long int ij, i, j;

  dim[0] = n; dim[1] = n;
  coef = (PyArrayObject *) PyArray_ZEROS(2, dim, NPY_DOUBLE, 0);
  if(!coef) {
    PyErr_SetString(PyExc_MemoryError, "Cannot create output array.");
    return NULL;
  }

#pragma omp parallel for private(i, j)
  for (ij = 0; ij < n*n; ij++) {
      i = ij/n;
      j = ij%n;
      (*(double *) PyArray_GETPTR2(coef, i, j)) = pearson2(d, i, j, l);
  }

  return coef;
}

// compute upper triangular part of the correlation matrix
// and store as a vector of length n*(n+1)/2
//
// original code by Rheinwalt
// adapted by Erik Scharwächter
//
// d: data array with n rows and l columns
// diagonal: (bool) include values on diagonal, default: 0
// mmap_arr: (bool) create temporary memory mapped file to hold the coefficient array (for large data sets)
// mmap_fd: pointer to an uninitialized (!) file descriptor for the mmap array, will be initialized
PyArrayObject *
pearson_triu(const double *d, unsigned long n, unsigned long l, int diagonal, int mmap_arr, int *mmap_fd) {
  PyArrayObject *coef;
  double *mmap_data;
  char mmap_filename[] = "tmpTriuCorrMat.XXXXXX";
  long int dim;
  int errcode;

  long i, k, o;
  double mk, sk, dk, h;
  double mi, si, sum;
  double *m, *s;
  double *c;

  if (diagonal)
    dim = n * (n + 1) / 2;
  else
    dim = n * (n - 1) / 2;

  if (!mmap_arr) {
    coef = (PyArrayObject *) PyArray_ZEROS(1, &dim, NPY_DOUBLE, 0);
    if(!coef) {
      PyErr_SetString(PyExc_MemoryError, "Cannot create output array.");
      return NULL;
    }
  } else {
    *mmap_fd = mkstemp(mmap_filename);
    if (*mmap_fd == -1) {
      perror(NULL);
      PyErr_SetString(PyExc_MemoryError, "Cannot create temporary file for memory map.");
      return NULL;
    }
    errcode = posix_fallocate(*mmap_fd, 0, dim*sizeof(double));
    if (errcode) {
      fprintf(stderr, "Failed allocating %ld bytes on disk.\n", dim*sizeof(double));
      fprintf(stderr, "%s\n", strerror(errcode));
      PyErr_SetString(PyExc_MemoryError, "Cannot resize temporary file.");
      return NULL;
    }
    mmap_data = mmap(NULL, dim*sizeof(double), (PROT_READ | PROT_WRITE), MAP_SHARED, *mmap_fd, 0);
    if (mmap_data == (void *) -1) {
      perror(NULL);
      PyErr_SetString(PyExc_MemoryError, "Cannot create memory mapped output array.");
      return NULL;
    }
    coef = (PyArrayObject *) PyArray_SimpleNewFromData(1, &dim, NPY_DOUBLE, mmap_data);
    if (!coef) {
      PyErr_SetString(PyExc_MemoryError, "Cannot create numpy array from memory map.");
      return NULL;
    }
  }

  /* mean and std */
  m = malloc(n * sizeof(double));
  s = malloc(n * sizeof(double));
  if (!m || !s) {
    PyErr_SetString(PyExc_MemoryError, "Cannot create mean and std arrays.");
    return NULL;
  }
#pragma omp parallel for private(k, h, mk, sk, dk)
  for (i = 0; i < n; i++) {
    mk = sk = 0;
    for (k = 0; k < l; k++) {
      dk = d[i*l + k];
      h = dk - mk;
      mk += h / (k + 1);
      sk += h * (dk - mk);
    }
    m[i] = mk;
    s[i] = sqrt(sk / (l - 1));
  }

  /* dot products */
  c = (double *) PyArray_DATA(coef);
#pragma omp parallel for private(k, mi, si, mk, sk, o, sum)
  for (i = 0; i < n; i++) {
    mi = m[i];
    si = s[i];
    for (k = i+(1-diagonal); k < n; k++) {
      mk = m[k];
      sk = s[k];
      sum = 0;
      for (o = 0; o < l; o++)
        sum += (d[i*l + o] - mi) * (d[k*l + o] - mk) / si / sk;
      if (diagonal)
        c[i*n-i*(i+1)/2+k] = sum / (l - 1);
      else
        c[i*(n-1)-i*(i+1)/2+k-1] = sum / (l - 1);
    }
  }
  free(m);
  free(s);

  return coef;
}

// find equivalence classes in a time series data set
//
// d: data set with n rows (time series) and l columns (time steps)
// alpha: transitivity threshold
// kappa: minimum cluster size
// max_nan: maximum number of nans within a pivot time series
PyArrayObject *
cluster(const double *d, unsigned long n, unsigned long l, double alpha, unsigned long kappa, unsigned long max_nan)
{
  unsigned long corr_count;
  unsigned long pivot, i, nan_count;
  double rho;
  llist_ul timeseries_l;
  llist_ul *clustermemb_pos_l;
  llist_ul *clustermemb_neg_l;
  llist_ul *noise_l;
  llist_ptr cluster_l;
  llist_item_ul *iter_ul, *iter_ul_next;
  llist_item_ptr *iter_ptr;

  PyArrayObject *membs = (PyArrayObject *) PyArray_ZEROS(1, (long int *) &n, NPY_LONG, 0);
  if(!membs) {
      PyErr_SetString(PyExc_MemoryError, "Cannot create output array.");
      return NULL;
  }

  // initialize time series index list
  llist_ul_init(&timeseries_l);
  for (i = 0; i < n; i++) {
    llist_ul_push_back(&timeseries_l, i);
  }

  // initialize cluster list
  llist_ptr_init(&cluster_l);

  // initialize noise cluster and add to cluster list (always at position 1)
  noise_l = (llist_ul *) malloc(sizeof(llist_ul));
  if (!noise_l) return NULL;
  llist_ul_init(noise_l);
  llist_ptr_push_back(&cluster_l, noise_l);

  // iterate over all time series until none is left
  corr_count = 0;
  while (llist_ul_size(&timeseries_l) > 0) {
    printf("\r% 9ld left...", llist_ul_size(&timeseries_l));
    pivot = llist_ul_back(&timeseries_l);

    // check if pivot contains too many nans to be considered a pivot
    nan_count = 0;
    for (i = 0; i < l; i++) {
      if (isnan(d[pivot*l+i])) nan_count++;
    }
    if (nan_count > max_nan) {
      // add pivot to noise cluster
      //printf("pivot %ld has too many nans\n", pivot);
      llist_ul_relink(timeseries_l.last, &timeseries_l, noise_l);
      continue;
    }

    // initialize positive and negative clusters
    clustermemb_pos_l = (llist_ul *) malloc(sizeof(llist_ul));
    if (!clustermemb_pos_l) return NULL;
    llist_ul_init(clustermemb_pos_l);
    clustermemb_neg_l = (llist_ul *) malloc(sizeof(llist_ul));
    if (!clustermemb_neg_l) return NULL;
    llist_ul_init(clustermemb_neg_l);

    // compute all correlations between pivot and remaining time series
    // and create positive and negative protoclusters
    iter_ul = timeseries_l.first;
    while (iter_ul != NULL) {
      iter_ul_next = iter_ul->next; // store successor before relinking
      rho = pearson2(d, pivot, iter_ul->data, l);
      corr_count++;
      if (isnan(rho)) {
        // TODO: we add the tested time series to the noise cluster, this might not be
        // a good idea if nan value occurs because there are no overlapping valid time steps
        // in pivot and tested time series
        //printf("rho=nan for pivot %ld and time series %ld\n", pivot, iter_ul->data);
        llist_ul_relink(iter_ul, &timeseries_l, noise_l);
      } else {
        if (rho >=  alpha) llist_ul_relink(iter_ul, &timeseries_l, clustermemb_pos_l);
        if (rho <= -alpha) llist_ul_relink(iter_ul, &timeseries_l, clustermemb_neg_l);
      }
      iter_ul = iter_ul_next;
    }

    // check whether protoclusters fulfill the minimium size constraints
    if (llist_ul_size(clustermemb_pos_l) >= kappa) {
      // add to final clustering
      llist_ptr_push_back(&cluster_l, clustermemb_pos_l);
    } else {
      // relink all time series to noise cluster
      llist_ul_relink_all(clustermemb_pos_l, noise_l);
      free(clustermemb_pos_l);
    }
    if (llist_ul_size(clustermemb_neg_l) >= kappa) {
      // add to final clustering
      llist_ptr_push_back(&cluster_l, clustermemb_neg_l);
    } else {
      // relink all time series to noise cluster
      llist_ul_relink_all(clustermemb_neg_l, noise_l);
      free(clustermemb_neg_l);
    }
  }
  printf("\rclustering finished with %ld correlation computations.\n", corr_count);

  // prepare output array with cluster assignments
  // skip noise cluster (membs id=0 during initialization)
  i = 1;
  iter_ptr = cluster_l.first->next;
  while (iter_ptr != NULL) {
    iter_ul = ((llist_ul *) iter_ptr->data)->first;
    while (iter_ul != NULL) {
      (*(long int *) PyArray_GETPTR1(membs, iter_ul->data)) = i;
      iter_ul = iter_ul->next;
    }
    llist_ul_destroy((llist_ul *) iter_ptr->data);
    free(iter_ptr->data);
    iter_ptr = iter_ptr->next;
    i++;
  }
  llist_ptr_destroy(&cluster_l);
  llist_ul_destroy(&timeseries_l);

  return membs;
}

// compute element-wise norms for evaluation:
// absolute deviation (L1), square root of sum of squared deviation (L2), maximum deviation (Lmax)
//
// d: data set with n rows and l columns (may be NULL if corr_triu specified)
// corr_triu: precomputed true correlations in triu array (may be NULL if d specified)
// corr_clus_triu: precomputed cluster correlations in triu array
// membs: cluster membership vector of length n
// n: number of time series (rows in d)
// l: number of time steps (columns in d)
// k: number of clusters
//
// output: numpy array [L1, L2, Lmax, nan_count], where nan_count is the number of time series
// pairs that were not considered because either their true or estimated correlation is NaN
PyArrayObject *
norms(const double *d, const double *corr_triu, const double *corr_clus_triu, const long *membs,
      unsigned long n, unsigned long l, unsigned long k) {
  double norm_l1 = 0.0;
  double norm_l2 = 0.0;
  double norm_max = 0.0;
  unsigned long nan_vals = 0;
  long i, j, ii, jj;
  int abort = 0;
  double corr_est, corr_tru;

  if((d == NULL) && (corr_triu == NULL)) {
    PyErr_SetString(PyExc_ValueError, "Either data matrix or precomputed correlations must be specified.");
    return NULL;
  }

  #pragma omp parallel for private(i, j, corr_tru, corr_est, ii, jj) \
                           reduction(+:norm_l1,norm_l2,nan_vals) \
                           reduction(max:norm_max)
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      // for error handling
      #pragma omp flush (abort)
      if (abort)
          continue;

      if ((membs[i] < 0) || (membs[i] >= k) || (membs[j] < 0) || (membs[j] >= k)) {
        PyErr_SetString(PyExc_ValueError,
              "Invalid cluster index (must have range 0, ..., k-1). Noise cluster 0 missing?");
        abort = 1;
        #pragma omp flush (abort)
      }

      if (corr_triu != NULL) { // use precomputed correlation
          corr_tru = corr_triu[i*n-(i*(i+1))/2+j]; // triu indexing (with diagonal)
      } else {
          corr_tru = pearson2(d, i, j, l);
      }
      ii = fminl(membs[i], membs[j]); // triu index formula below for ii<jj
      jj = fmaxl(membs[i], membs[j]);
      corr_est = corr_clus_triu[ii*k-ii*(ii+1)/2+jj]; // triu indexing (with diagonal)

      if (isnan(corr_tru) || isnan(corr_est)) {
        nan_vals++;
      } else {
        norm_l1 += fabs(corr_tru-corr_est);
        norm_l2 += (corr_tru-corr_est)*(corr_tru-corr_est);
        if (norm_max < fabs(corr_tru-corr_est)) {
            norm_max = fabs(corr_tru-corr_est);
        }
      }
    }
  }
  norm_l2 = sqrt(norm_l2);
  if (abort) {
      return NULL;
  }

  // prepare output
  PyArrayObject *norm_vec;
  long int dim = 4;
  norm_vec = (PyArrayObject *) PyArray_ZEROS(1, &dim, NPY_DOUBLE, 0);
  if(!norm_vec) {
    PyErr_SetString(PyExc_MemoryError, "Cannot create output array.");
    return NULL;
  }
  *((double *) PyArray_GETPTR1(norm_vec, 0)) = norm_l1;
  *((double *) PyArray_GETPTR1(norm_vec, 1)) = norm_l2;
  *((double *) PyArray_GETPTR1(norm_vec, 2)) = norm_max;
  *((double *) PyArray_GETPTR1(norm_vec, 3)) = (double) nan_vals;
  return norm_vec;
}

/* ######################## PYTHON BINDINGS ######################## */

static PyObject *
COREQ_Norms(PyObject *self, PyObject* args) {
  PyObject *arg1, *arg2, *arg3;
  PyArrayObject *input_arr, *cluster_corr, *membs, *norm_vec;
  int precomputed;

  precomputed = 0;
  if (!PyArg_ParseTuple(args, "OOO|i", &arg1, &arg2, &arg3, &precomputed))
    return NULL;

  if (precomputed) {
      // shape: (N*(N+1)/2,)
      input_arr = (PyArrayObject *) PyArray_ContiguousFromObject(arg1, NPY_DOUBLE, 1, 1);
  } else {
      // shape: (N,D)
      input_arr = (PyArrayObject *) PyArray_ContiguousFromObject(arg1, NPY_DOUBLE, 2, 2);
  }
  if (!input_arr) return NULL;
  cluster_corr = (PyArrayObject *) PyArray_ContiguousFromObject(arg2, NPY_DOUBLE, 1, 1);
  if (!cluster_corr) return NULL;
  membs = (PyArrayObject *) PyArray_ContiguousFromObject(arg3, NPY_LONG, 1, 1);
  if (!membs) return NULL;

  if (precomputed) {
    norm_vec = norms(NULL, (double *)PyArray_DATA(input_arr), (double *)PyArray_DATA(cluster_corr), (long int *)PyArray_DATA(membs),
                  PyArray_DIM(membs, 0), 0, -1/2.+sqrt(1/4.+2.*PyArray_DIM(cluster_corr, 0)));
  } else {
    norm_vec = norms((double *)PyArray_DATA(input_arr), NULL, (double *)PyArray_DATA(cluster_corr), (long int *)PyArray_DATA(membs),
                  PyArray_DIM(membs, 0), PyArray_DIM(input_arr, 1), -1/2.+sqrt(1/4.+2.*PyArray_DIM(cluster_corr, 0)));
  }

  Py_DECREF(input_arr);
  Py_DECREF(cluster_corr);
  Py_DECREF(membs);
  return PyArray_Return(norm_vec);
}

static PyObject *
COREQ_Pearson(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *coef;

  if(!PyArg_ParseTuple(args, "O", &arg))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if(!data)
    return NULL;

  coef = pearson((double *)PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1));

  Py_DECREF(data);
  return PyArray_Return(coef);
}

/* TODO: mmap_fd is never closed and file is forgotten -> unnecessary hdd consumption */
static PyObject *
COREQ_PearsonTriu(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *coef;
  int diagonal, mmap_arr;
  int mmap_fd;

  diagonal = 0;
  mmap_arr = 0;
  if(!PyArg_ParseTuple(args, "O|ii", &arg, &diagonal, &mmap_arr))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if(!data)
    return NULL;

  coef = pearson_triu((double *)PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1),
      diagonal, mmap_arr, &mmap_fd);

  Py_DECREF(data);
  return PyArray_Return(coef);
}

static PyObject *
COREQ_Cluster(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *clus;
  double alpha;
  unsigned long kappa, max_nan;

  if(!PyArg_ParseTuple(args, "Odkk", &arg, &alpha, &kappa, &max_nan))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if(!data)
    return NULL;

  clus = cluster((double *)PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1),
      alpha, kappa, max_nan);

  Py_DECREF(data);
  return PyArray_Return(clus);
}

static PyMethodDef COREQ_methods[] = {
  {"Pearson", COREQ_Pearson, METH_VARARGS,
   "corr = Pearson(data)\n\n...\n"},
  {"PearsonTriu", COREQ_PearsonTriu, METH_VARARGS,
   "triu_corr = PearsonTriu(data, diagonal=False, mmap=0)\n\nReturn Pearson product-moment correlation coefficients.\n\nParameters\n----------\ndata : array_like\nA 2-D array containing multiple variables and observations. Each row of `data` represents a variable, and each column a single observation of all those variables.\n\nReturns\n-------\ntriu_corr : ndarray\nThe upper triangle of the correlation coefficient matrix of the variables.\n"},
  {"Cluster", COREQ_Cluster, METH_VARARGS,
   "labels = Cluster(data, alpha, kappa, max_nan)\n\n...\n"},
  {"Norms", COREQ_Norms, METH_VARARGS,
   "norm_vec = Norms(input_array, cluster_corr, membs, precomputed=False)\n\nIf precomputed is False (default), input_array is interpreted as a data matrix with N rows and D columns. Otherwise, it is interpreted as a triu correlation matrix of size N*(N+1)/2.\n"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_INIT(name) void init##name(void)
  #define MOD_SUCCESS_VAL(val)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(coreq)
{
  PyObject *m;
  MOD_DEF(m, "coreq", "Build equivalence classes for low redundancy correlation estimation.",
          COREQ_methods)
  if (m == NULL)
    return;
  import_array(); // numpy import
  return MOD_SUCCESS_VAL(m);
}

int
main(int argc, char **argv) {
  Py_SetProgramName(argv[0]);
  Py_Initialize();
    PyImport_ImportModule("coreq");
  Py_Exit(0);
  return 0;
}
