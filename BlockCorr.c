#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "list.h"

#define VERSION "0.3"
#include "BlockCorr.h"

// compute pearson correlation coefficient between time series at positions i1 and i2 in d (of length l)
// NOTE: result may be nan, if the variance of any of the time series is zero, or if
// any of the time series contains nans
double pearson2(const double *d, const unsigned long i, const unsigned long j, const unsigned long l) {
  unsigned int k;
  double sum_i = 0.0, sum_j = 0.0, sum_ii = 0.0, sum_jj = 0.0, sum_ij = 0.0;
#pragma omp simd
  for (k = 0; k < l; k++) {
    sum_i += d[i*l+k];
    sum_j += d[j*l+k];
    sum_ii += d[i*l+k]*d[i*l+k];
    sum_jj += d[j*l+k]*d[j*l+k];
    sum_ij += d[i*l+k]*d[j*l+k];
  }
  return (l*sum_ij-sum_i*sum_j)/sqrt((l*sum_ii-sum_i*sum_i)*(l*sum_jj-sum_j*sum_j));
}

// compute n-by-n correlation matrix for complete data set d with n rows and l columns
double *pearson(const double *d, unsigned long n, unsigned long l) {
  long int ij, i, j;
  double *coef;

  // allocate memory
  coef = calloc(n*n, sizeof (double));
  if (!coef) {
    return NULL;
  }

#pragma omp parallel for private(i, j)
  for (ij = 0; ij < n*n; ij++) {
      i = ij/n;
      j = ij%n;
      if (i > j) continue;
      coef[i*n+j] = pearson2(d, i, j, l);
      coef[j*n+i] = coef[i*n+j];
  }

  return coef;
}

// compute upper triangular part of the correlation matrix
// and store as a vector of length n*(n+1)/2
//
// original code by Aljoscha Rheinwalt
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

#ifdef DEBUG
    llist_item_ul *tmp_iter1;
    llist_item_ul *tmp_iter2;
    tmp_iter1 = clustermemb_pos_l->first;
    while (tmp_iter1) {
      tmp_iter2 = tmp_iter1;
      while (tmp_iter2) {
        printf("pos %ld %ld %.2f\n", tmp_iter1->data, tmp_iter2->data, pearson2(d, tmp_iter1->data, tmp_iter2->data, l));
        tmp_iter2 = tmp_iter2->next;
      }
      tmp_iter1 = tmp_iter1->next;
    }
    tmp_iter1 = clustermemb_neg_l->first;
    while (tmp_iter1) {
      tmp_iter2 = tmp_iter1;
      while (tmp_iter2) {
        printf("neg %ld %ld %.2f\n", tmp_iter1->data, tmp_iter2->data, pearson2(d, tmp_iter1->data, tmp_iter2->data, l));
        tmp_iter2 = tmp_iter2->next;
      }
      tmp_iter1 = tmp_iter1->next;
    }
#endif

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
#ifdef DEBUG
      printf("%ld -> %ld\n", iter_ul->data, i);
#endif
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

// assumes that the key really is somewhere in the array
unsigned long binary_search_ul(unsigned long key, unsigned long *arr, unsigned long len) {
  unsigned long liml, limr, rpos;
  liml = 0;
  limr = len-1;
  do {
    rpos = (liml+limr)/2;
    if (arr[rpos] < key) {
      liml = rpos + 1;
    } else if (arr[rpos] > key) {
      limr = rpos - 1;
    } else {
      break; // found it
    }
  } while (liml <= limr);
  return rpos;
}

// COREQ++
// find equivalence classes in a time series data set and estimate correlations
// NOTE: no kappa, no noise cluster estimation, no negative clusters, no NaN handling
//
// INPUT
// d: data set with n rows (time series) and l columns (time steps)
// n, l: see above
// alpha: transitivity threshold
// est_strat: estimation strategy (pivot-based, average-based)
//
// OUTPUT
// membs: uninitialized pointer to array for class memberships
// pivots: uninitialized pointer to array for pivot indices
// cluster_corrs: uninitialized pointer to array for class correlations (upper triangular indexing)
// n_clus: total number of resulting clusters
// corr_comps: total number of correlation computations required for clustering/estimation
long int *
coreqPP(const double *d, unsigned long n, unsigned long l, double alpha, coreq_estimation_strategy_t est_strat,
    long int **membs, long int **pivots, double **cluster_corrs,
    long int *n_clus, long int *corr_comps)
{
  unsigned long pivot, remaining, i, j, k, rpos, sample_size;
  double rho;
  llist_ul timeseries_l; // holds all unprocessed time series
  llist_ul pivot_l; // holds all pivot objects selected so far
  llist_ul *clustermemb_l; // holds all time series assigned to a cluster
  llist_ptr cluster_l; // holds all clusters
  llist_ptr correlations_idx_l; // holds corrs between all pivots and time series (indices)
  llist_ptr correlations_val_l; // holds corrs between all pivots and time series (correlations)
  llist_ul  correlations_cnt_l; // holds the number of entries in the previous lists
  llist_item_ul *iter_ul, *iter_ul_next;
  llist_item_ptr *iter_ptr, *iter_idx, *iter_val;
  unsigned long **cluster_arr; // hold all clusters with their members
  unsigned long *cluster_size_arr; // hold all cluster sizes

  *membs = calloc(n, sizeof(long int));
  if (!*membs) return NULL;

  // initialize time series index list
  llist_ul_init(&timeseries_l);
  for (i = 0; i < n; i++) {
    llist_ul_push_back(&timeseries_l, i);
  }

  // initialize lists
  llist_ptr_init(&cluster_l);
  llist_ptr_init(&correlations_idx_l);
  llist_ptr_init(&correlations_val_l);
  llist_ul_init(&correlations_cnt_l);
  llist_ul_init(&pivot_l);

  // iterate over all time series until none is left
  *corr_comps = 0;
  while (llist_ul_size(&timeseries_l) > 0) {
    remaining = llist_ul_size(&timeseries_l);
    printf("\r% 9ld left...", remaining);

    // select pivot time series and create its correlation container
    pivot = llist_ul_front(&timeseries_l);
    llist_ul_push_back(&pivot_l, pivot);
    llist_ptr_push_back(&correlations_idx_l, calloc(remaining, sizeof (unsigned long)));
    llist_ptr_push_back(&correlations_val_l, calloc(remaining, sizeof (double)));
    llist_ul_push_back(&correlations_cnt_l, remaining);

    // initialize cluster container
    clustermemb_l = (llist_ul *) malloc(sizeof (llist_ul));
    if (!clustermemb_l) return NULL;
    llist_ul_init(clustermemb_l);

    // compute all correlations between pivot and remaining time series
    iter_ul = timeseries_l.first;
    i = 0;
    while (iter_ul != NULL) {
      iter_ul_next = iter_ul->next; // store successor before relinking

      // compute correlation
      rho = pearson2(d, pivot, iter_ul->data, l);
      (*corr_comps)++;
      ((unsigned long *) llist_ptr_back(&correlations_idx_l))[i] = iter_ul->data;
      ((double *) llist_ptr_back(&correlations_val_l))[i] = rho;

      // add time series to cluster
      if (rho >= alpha) {
        llist_ul_relink(iter_ul, &timeseries_l, clustermemb_l);
      }

      iter_ul = iter_ul_next;
      i++;
    }

    // add to final clustering
    llist_ptr_push_back(&cluster_l, clustermemb_l);
  }
  *n_clus = llist_ul_size(&pivot_l);
  printf("\rclustering finished with %ld correlation computations --- %ld clusters detected\n", *corr_comps, *n_clus);

  // prepare output array with cluster assignments
  // and buffer all clusters in cluster_arr for O(1) access to members
  cluster_arr = calloc(*n_clus, sizeof (unsigned long *));
  cluster_size_arr = calloc(*n_clus, sizeof (unsigned long));
  i = 0;
  iter_ptr = cluster_l.first;
  while (iter_ptr != NULL) {
    cluster_arr[i] = calloc(llist_ul_size((llist_ul *) iter_ptr->data), sizeof (unsigned long));
    cluster_size_arr[i] = llist_ul_size((llist_ul *) iter_ptr->data);
    j = 0;
    iter_ul = ((llist_ul *) iter_ptr->data)->first;
    while (iter_ul != NULL) {
      cluster_arr[i][j] = iter_ul->data;
      (*membs)[iter_ul->data] = i;
      iter_ul = iter_ul->next;
      j++;
    }
    llist_ul_destroy((llist_ul *) iter_ptr->data);
    free(iter_ptr->data);
    iter_ptr = iter_ptr->next;
    i++;
  }

  // prepare output array with pivots
  *pivots = calloc(*n_clus, sizeof (long int));
  i = 0;
  iter_ul = pivot_l.first;
  while (iter_ul != NULL) {
    (*pivots)[i] = iter_ul->data;
    iter_ul = iter_ul->next;
    i++;
  }

  // prepare output array with correlation estimates in O(K*(K+1)/2 * log2(N) * log2(N))
  // NOTE: we use binary search to look up the precomputed pivot-time series correlations;
  // no additional correlation computations are necessary, which would require O(K*(K+1)/2 * T * log2(N))
  *cluster_corrs = calloc((*n_clus)*(*n_clus+1)/2, sizeof (double));
  iter_idx = correlations_idx_l.first;
  iter_val = correlations_val_l.first;
  iter_ul = correlations_cnt_l.first;
  for (i = 0; i < (*n_clus); i++) { // loop over all pairs of pivots
    for (j = i; j < (*n_clus); j++) {
      switch (est_strat) {
        case COREQ_PIVOT:
          // search for pivot j in the correlation index list of pivot i
          rpos = binary_search_ul((*pivots)[j], ((unsigned long *) iter_idx->data), iter_ul->data);
          //printf("p%lu=ts%lu@loc%lu(ts%lu):%.2f ", j, (*pivots)[j], rpos, ((unsigned long *) iter_idx->data)[rpos], ((double *) iter_val->data)[rpos]);
          // retrieve pivot i-j correlation from position correlation value list at position rpos
          (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] = ((double *) iter_val->data)[rpos];
          break;
        case COREQ_PIVOT_GUARANTEE:
          // same as above, but with scaling alpha-dependent scaling factor
          rpos = binary_search_ul((*pivots)[j], ((unsigned long *) iter_idx->data), iter_ul->data);
          (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] = 0.5*(1.0+alpha*alpha) * ((double *) iter_val->data)[rpos];
          break;
        case COREQ_AVERAGE:
          // sample log2(N_k) precomputed correlations and use average as estimate
          sample_size = fmax(1,ceil(log2(cluster_size_arr[j])));
          (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] = 0;
          for (k = 0; k < sample_size; k++) {
            // sample random member of cluster j
            unsigned long sample = rand() % cluster_size_arr[j];
            rpos = binary_search_ul(cluster_arr[j][sample], ((unsigned long *) iter_idx->data), iter_ul->data);
            (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] += ((double *) iter_val->data)[rpos];
          }
          (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] /= sample_size;
          break;
        default:
          return NULL;
      }
    }

    iter_idx = iter_idx->next;
    iter_val = iter_val->next;
    iter_ul = iter_ul->next;
  }

  // destroy correlation containers
  iter_ptr = correlations_idx_l.first;
  while (iter_ptr != NULL) {
    free(iter_ptr->data);
    iter_ptr = iter_ptr->next;
  }
  iter_ptr = correlations_val_l.first;
  while (iter_ptr != NULL) {
    free(iter_ptr->data);
    iter_ptr = iter_ptr->next;
  }

  // destroy cluster buffer array
  for (i = 0; i < *n_clus; i++) {
    free(cluster_arr[i]);
  }
  free(cluster_arr);
  free(cluster_size_arr);

  llist_ptr_destroy(&cluster_l);
  llist_ptr_destroy(&correlations_idx_l);
  llist_ptr_destroy(&correlations_val_l);
  llist_ul_destroy(&correlations_cnt_l);
  llist_ul_destroy(&timeseries_l);
  llist_ul_destroy(&pivot_l);

  return *membs;
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
#ifdef DEBUG
        if (fabs(corr_tru-corr_est) > norm_max) {
          printf("%ld\t%ld\t%ld | %ld\t%ld\t%ld | %.2f\t%.2f\n", i, j, i*n-i*(i+1)/2+j,
              membs[i], membs[j], ii*k-(ii*(ii+1))/2+jj, corr_est, corr_tru);
        }
#endif
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
BlockCorr_Norms(PyObject *self, PyObject* args) {
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
BlockCorr_Pearson(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *coef_py;
  double *coef;

  if(!PyArg_ParseTuple(args, "O", &arg))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if(!data)
    return NULL;

  coef = pearson((double *)PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1));
  if (!coef)
    return NULL;

  long int dims[2] = {PyArray_DIM(data, 0), PyArray_DIM(data, 0)};
  coef_py = (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, coef);

  Py_DECREF(data);
  return PyArray_Return(coef_py);
}

/* TODO: mmap_fd is never closed and file is forgotten -> unnecessary hdd consumption */
static PyObject *
BlockCorr_PearsonTriu(PyObject *self, PyObject* args) {
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
BlockCorr_Cluster(PyObject *self, PyObject* args) {
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

static PyObject *
BlockCorr_COREQpp(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *membs_arr, *pivots_arr, *cluster_corrs_arr;
  long int *membs, *pivots;
  double *cluster_corrs;
  long int corr_comps, n_clus, n_corrs;
  double alpha;
  unsigned long n, l;
  coreq_estimation_strategy_t est_strat;

  if(!PyArg_ParseTuple(args, "Okd", &arg, &est_strat, &alpha))
    return NULL;

  // run COREQ++
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if (!data) return NULL;
  n = PyArray_DIM(data, 0);
  l = PyArray_DIM(data, 1);
  if (!coreqPP((double *)PyArray_DATA(data), n, l, alpha, est_strat, &membs, &pivots, &cluster_corrs, &n_clus, &corr_comps)) {
      PyErr_SetString(PyExc_MemoryError, "Cannot create output array.");
      return NULL;
  }
  Py_DECREF(data);

  // prepare Python output (cluster assignments)
  membs_arr = (PyArrayObject *) PyArray_SimpleNewFromData(1, (long int *) &n, NPY_LONG, membs);
  if (!membs_arr) {
      PyErr_SetString(PyExc_MemoryError, "Cannot create Python reference to output array (memberships).");
      return NULL;
  }

  // prepare Python output (pivot choices)
  pivots_arr = (PyArrayObject *) PyArray_SimpleNewFromData(1, &n_clus, NPY_LONG, pivots);
  if (!pivots_arr) {
      PyErr_SetString(PyExc_MemoryError, "Cannot create Python reference to output array (pivots).");
      return NULL;
  }

  // prepare Python output (cluster correlations)
  n_corrs = n_clus*(n_clus+1)/2;
  cluster_corrs_arr = (PyArrayObject *) PyArray_SimpleNewFromData(1, &n_corrs, NPY_DOUBLE, cluster_corrs);
  if (!pivots_arr) {
      PyErr_SetString(PyExc_MemoryError, "Cannot create Python reference to output array (correlations).");
      return NULL;
  }

  return Py_BuildValue("OOOl", (PyObject *) membs_arr, (PyObject *) pivots_arr, (PyObject *) cluster_corrs_arr, corr_comps);
}

static PyMethodDef BlockCorr_methods[] = {
  {"Pearson", BlockCorr_Pearson, METH_VARARGS,
   "corr = Pearson(data)\n\n...\n"},
  {"PearsonTriu", BlockCorr_PearsonTriu, METH_VARARGS,
   "triu_corr = PearsonTriu(data, diagonal=False, mmap=0)\n\nReturn Pearson product-moment correlation coefficients.\n\nParameters\n----------\ndata : array_like\nA 2-D array containing multiple variables and observations. Each row of `data` represents a variable, and each column a single observation of all those variables.\n\nReturns\n-------\ntriu_corr : ndarray\nThe upper triangle of the correlation coefficient matrix of the variables.\n"},
  {"Cluster", BlockCorr_Cluster, METH_VARARGS,
   "labels = Cluster(data, alpha, kappa, max_nan)\n\n...\n"},
  {"COREQpp", BlockCorr_COREQpp, METH_VARARGS,
   "(labels, pivots, pivot_corr_triu, computations) = COREQpp(data, alpha, estimation_strategy)\n\n...\n"},
  {"Norms", BlockCorr_Norms, METH_VARARGS,
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

MOD_INIT(BlockCorr)
{
  PyObject *m;
  MOD_DEF(m, "BlockCorr", "Block matrix estimation for correlation coefficients.",
          BlockCorr_methods)
  if (m == NULL)
    return;
  import_array(); // numpy import
  if (PyModule_AddIntConstant(m, "ESTIMATE_PIVOT", COREQ_PIVOT))
    return;
  if (PyModule_AddIntConstant(m, "ESTIMATE_PIVOT_GUARANTEE", COREQ_PIVOT_GUARANTEE))
    return;
  if (PyModule_AddIntConstant(m, "ESTIMATE_AVERAGE", COREQ_AVERAGE))
    return;
  return MOD_SUCCESS_VAL(m);
}

int
main(int argc, char **argv) {
  Py_SetProgramName(argv[0]);
  Py_Initialize();
    PyImport_ImportModule("BlockCorr");
  Py_Exit(0);
  return 0;
}

