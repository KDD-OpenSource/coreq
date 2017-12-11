#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "BlockCorr.h"
#include "numpy/arrayobject.h"

static PyObject *
BlockCorr_Loss(PyObject *self, PyObject* args) {
  PyObject *arg1, *arg2, *arg3;
  PyArrayObject *input_arr, *cluster_corr, *membs;
  int precomputed, success;
  double loss_abs, loss_sq, loss_max;
  long elements;

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
  if (!input_arr) {
    return NULL;
  }

  cluster_corr = (PyArrayObject *) PyArray_ContiguousFromObject(arg2, NPY_DOUBLE, 1, 1);
  if (!cluster_corr) {
    Py_DECREF(input_arr);
    return NULL;
  }

  membs = (PyArrayObject *) PyArray_ContiguousFromObject(arg3, NPY_LONG, 1, 1);
  if (!membs) {
    Py_DECREF(input_arr);
    Py_DECREF(cluster_corr);
    return NULL;
  }

  if (precomputed) {
    success = compute_loss(NULL, (double *) PyArray_DATA(input_arr), (double *) PyArray_DATA(cluster_corr),
                  (long int *) PyArray_DATA(membs),
                  PyArray_DIM(membs, 0), 0, -1/2.+sqrt(1/4.+2.*PyArray_DIM(cluster_corr, 0)),
                  &loss_abs, &loss_sq, &loss_max, &elements);
  } else {
    success = compute_loss((double *) PyArray_DATA(input_arr), NULL, (double *) PyArray_DATA(cluster_corr),
                  (long int *) PyArray_DATA(membs),
                  PyArray_DIM(membs, 0), PyArray_DIM(input_arr, 1), -1/2.+sqrt(1/4.+2.*PyArray_DIM(cluster_corr, 0)),
                  &loss_abs, &loss_sq, &loss_max, &elements);
  }

  Py_DECREF(input_arr);
  Py_DECREF(cluster_corr);
  Py_DECREF(membs);

  switch (success) {
    case 0:
      return Py_BuildValue("dddl", loss_abs, loss_sq, loss_max, elements);
    case -1:
      PyErr_SetString(PyExc_ValueError, "Specify either the input data or a precomputed correlation matrix");
      return NULL;
    case -2:
      PyErr_SetString(PyExc_ValueError, "Invalid cluster id in membership vector, range [0...K]?");
      return NULL;
    default:
      return NULL;
  }
}

static PyObject *
BlockCorr_Pearson(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *coef_arr;
  double *coef;

  if (!PyArg_ParseTuple(args, "O", &arg))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if (!data)
    return NULL;

  coef = pearson((double *) PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1));
  if (!coef) {
    PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for correlation matrix");
    Py_DECREF(data);
    return NULL;
  }

  long int dims[2] = {PyArray_DIM(data, 0), PyArray_DIM(data, 0)};
  coef_arr = (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, coef);
  if (!coef_arr) {
      Py_DECREF(data);
      return NULL;
  }

  Py_DECREF(data);
  return PyArray_Return(coef_arr);
}

static PyObject *
BlockCorr_PearsonTriu(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *coef_arr;
  double *coef;

  if (!PyArg_ParseTuple(args, "O", &arg))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if (!data)
    return NULL;

  coef = pearson_triu((double *) PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1));
  if (!coef) {
    PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for correlation matrix");
    Py_DECREF(data);
    return NULL;
  }

  long int dims[1] = {PyArray_DIM(data, 0)*(PyArray_DIM(data, 0)+1)/2};
  coef_arr = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, coef);
  if (!coef_arr) {
    Py_DECREF(data);
    return NULL;
  }

  Py_DECREF(data);
  return PyArray_Return(coef_arr);
}

static PyObject *
BlockCorr_Cluster(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *clus_arr;
  double alpha;
  long kappa, max_nan;
  long *clus;

  if (!PyArg_ParseTuple(args, "Odll", &arg, &alpha, &kappa, &max_nan))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if (!data)
    return NULL;

  clus = cluster((double *)PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1),
      alpha, kappa, max_nan);
  if (!clus) {
    PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for clustering");
    Py_DECREF(data);
    return NULL;
  }

  long int dims[1] = {PyArray_DIM(data, 0)};
  clus_arr = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_LONG, clus);
  if (!clus_arr) {
    Py_DECREF(data);
    return NULL;
  }

  Py_DECREF(data);
  return PyArray_Return(clus_arr);
}

static PyObject *
BlockCorr_COREQ(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *membs_arr, *pivots_arr, *cluster_corrs_arr;
  long int *membs, *pivots;
  double *cluster_corrs;
  long int corr_comps, n_clus, n_corrs;
  double alpha;
  long n, l;
  coreq_estimation_strategy_t est_strat;

  if (!PyArg_ParseTuple(args, "Oid", &arg, &est_strat, &alpha))
    return NULL;

  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg, NPY_DOUBLE, 2, 2);
  if (!data) return NULL;
  n = PyArray_DIM(data, 0);
  l = PyArray_DIM(data, 1);
  if (!coreq((double *)PyArray_DATA(data), n, l, alpha, est_strat, &membs, &pivots, &cluster_corrs, &n_clus, &corr_comps)) {
    PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for clustering");
    Py_DECREF(data);
    return NULL;
  }
  Py_DECREF(data);

  // prepare Python output (cluster assignments)
  membs_arr = (PyArrayObject *) PyArray_SimpleNewFromData(1, (long int *) &n, NPY_LONG, membs);
  if (!membs_arr) {
    Py_DECREF(data);
    return NULL;
  }

  // prepare Python output (pivot choices)
  pivots_arr = (PyArrayObject *) PyArray_SimpleNewFromData(1, &n_clus, NPY_LONG, pivots);
  if (!pivots_arr) {
    Py_DECREF(data);
    return NULL;
  }

  // prepare Python output (cluster correlations)
  n_corrs = n_clus*(n_clus+1)/2;
  cluster_corrs_arr = (PyArrayObject *) PyArray_SimpleNewFromData(1, &n_corrs, NPY_DOUBLE, cluster_corrs);
  if (!pivots_arr) {
    Py_DECREF(data);
    return NULL;
  }

  return Py_BuildValue("OOOl", (PyObject *) membs_arr, (PyObject *) pivots_arr, (PyObject *) cluster_corrs_arr, corr_comps);
}

static PyMethodDef BlockCorr_methods[] = {
  {"Pearson", BlockCorr_Pearson, METH_VARARGS,
   "corr = Pearson(data)"},
  {"PearsonTriu", BlockCorr_PearsonTriu, METH_VARARGS,
   "triu_corr = PearsonTriu(data, diagonal=False)"},
  {"Cluster", BlockCorr_Cluster, METH_VARARGS,
   "labels = Cluster(data, alpha, kappa, max_nan)"},
  {"COREQ", BlockCorr_COREQ, METH_VARARGS,
   "(labels, pivots, pivot_corr_triu, computations) = COREQ(data, estimation_strategy, alpha)"},
  {"Loss", BlockCorr_Loss, METH_VARARGS,
   "(abs, sq, max, elems) = Loss(input_array, cluster_corr, membs, precomputed=False)"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef BlockCorr_module = {
  PyModuleDef_HEAD_INIT,
  "BlockCorr", // name
  NULL,        // module documentation
  -1,          // size of per-interpreter state of the module
               // or -1 if the module keeps state in global variables.
  BlockCorr_methods
};
#endif

#if PY_MAJOR_VERSION >= 3
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_RETURN(val) return val
#else
  #define MOD_INIT(name) void init##name(void)
  #define MOD_RETURN(val) return
#endif

MOD_INIT(BlockCorr) {
#if PY_MAJOR_VERSION < 3
  PyObject *m = Py_InitModule("BlockCorr", BlockCorr_methods);
#else
  PyObject *m = PyModule_Create(&BlockCorr_module);
#endif
  if (m == NULL)
    MOD_RETURN(NULL);

  if (PyModule_AddIntConstant(m, "ESTIMATE_PIVOT", COREQ_PIVOT))
    MOD_RETURN(NULL);
  if (PyModule_AddIntConstant(m, "ESTIMATE_PIVOT_GUARANTEE", COREQ_PIVOT_GUARANTEE))
    MOD_RETURN(NULL);
  if (PyModule_AddIntConstant(m, "ESTIMATE_AVERAGE", COREQ_AVERAGE))
    MOD_RETURN(NULL);

  import_array(); // numpy import
  MOD_RETURN(m);
}

