#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "list.h"
#include "BlockCorr.h"

// compute pearson correlation coefficient between time series at positions i1 and i2 in d (of length l)
// NOTE: result may be nan, if the variance of any of the time series is zero, or if
// any of the time series contains nans
double pearson2(const double *d, const long int i, const long int j, const long int l) {
  long int k;
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
double *pearson(const double *d, long int n, long int l) {
  double *sums = (double *) calloc(n, sizeof (double));
  double *sumsqs = (double *) calloc(n, sizeof (double));
  double *sqsums = (double *) calloc(n, sizeof (double));
  double *coef = (double *) calloc(n*n, sizeof (double));
  if (!coef || !sums || !sumsqs || !sqsums) return NULL;

#pragma omp parallel for
  for (long int i = 0; i < n; i++) {
#pragma omp simd
    for (long k = 0; k < l; k++) {
      sums[i] += d[i*l+k];
      sumsqs[i] += d[i*l+k]*d[i*l+k];
    }
  }

#pragma omp simd
  for (long int i = 0; i < n; i++) {
    sqsums[i] = sums[i]*sums[i];
  }

#pragma omp parallel for
  for (long int ij = 0; ij < n*(n-1)/2; ij++) {
    double sum_ij = 0.0;
    long int i = ij / n;
    long int j = ij % n;
    if (j <= i) {
      i = n - i - 2;
      j = n - j - 1;
    }
#pragma omp simd
    for (long int k = 0; k < l; k++) {
      sum_ij += d[i*l+k]*d[j*l+k];
    }
    coef[i*n+j] = (l*sum_ij-sums[i]*sums[j])/sqrt((l*sumsqs[i]-sqsums[i])*(l*sumsqs[j]-sqsums[j]));
    coef[j*n+i] = coef[i*n+j];
  }

#pragma omp simd
  for (long int i = 0; i < n; i++) {
    coef[i*n+i] = 1.0;
  }

  free(sums);
  free(sumsqs);
  free(sqsums);
  return coef;
}

// compute upper triangular part of the correlation matrix
// and store as a vector of length n*(n+1)/2
double *
pearson_triu(const double *d, long int n, long int l) {
  long int i, j, k;
  double *sums = (double *) calloc(n, sizeof (double));
  double *sumsqs = (double *) calloc(n, sizeof (double));
  double *coef = (double *) calloc(n*(n+1)/2, sizeof (double));
  double sum_ij;
  if (!coef) return NULL;

#pragma omp parallel for
  for (i = 0; i < n; i++) {
#pragma omp simd
    for (k = 0; k < l; k++) {
      sums[i] += d[i*l+k];
      sumsqs[i] += d[i*l+k]*d[i*l+k];
    }
  }

#pragma omp parallel for collapse(2) private (sum_ij) schedule(dynamic)
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (i > j) continue;
      sum_ij = 0.0;
#pragma omp simd
      for (k = 0; k < l; k++) {
        sum_ij += d[i*l+k]*d[j*l+k];
      }
      coef[i*n-i*(i+1)/2+j] = (l*sum_ij-sums[i]*sums[j])/sqrt((l*sumsqs[i]-sums[i]*sums[i])*(l*sumsqs[j]-sums[j]*sums[j]));
    }
  }

  free(sums);
  free(sumsqs);
  return coef;
}

// find equivalence classes in a time series data set
//
// d: data set with n rows (time series) and l columns (time steps)
// alpha: transitivity threshold
// kappa: minimum cluster size
// max_nan: maximum number of nans within a pivot time series
long int *cluster(const double *d, long int n, long int l, double alpha, long int kappa, long int max_nan)
{
  long int corr_count, pivot, i, nan_count;
  long int *membs;
  double rho;
  llist_li timeseries_l;
  llist_li *clustermemb_pos_l;
  llist_li *clustermemb_neg_l;
  llist_li *noise_l;
  llist_ptr cluster_l;
  llist_item_li *iter_li, *iter_li_next;
  llist_item_ptr *iter_ptr;

  membs = (long int *) calloc(n, sizeof (long int));
  if (!membs) {
      return NULL;
  }

  // initialize time series index list
  llist_li_init(&timeseries_l);
  for (i = 0; i < n; i++) {
    llist_li_push_back(&timeseries_l, i);
  }

  // initialize cluster list
  llist_ptr_init(&cluster_l);

  // initialize noise cluster and add to cluster list (always at position 1)
  noise_l = (llist_li *) malloc(sizeof(llist_li));
  if (!noise_l) return NULL;
  llist_li_init(noise_l);
  llist_ptr_push_back(&cluster_l, noise_l);

  // iterate over all time series until none is left
  corr_count = 0;
  while (llist_li_size(&timeseries_l) > 0) {
    printf("\r% 9ld left...", llist_li_size(&timeseries_l));
    pivot = llist_li_front(&timeseries_l);

    // check if pivot contains too many nans to be considered a pivot
    nan_count = 0;
    for (i = 0; i < l; i++)
      if (isnan(d[pivot*l+i]))
        nan_count++;
    if (nan_count > max_nan) {
      // add pivot to noise cluster
      llist_li_relink(timeseries_l.last, &timeseries_l, noise_l);
      continue;
    }

    // initialize positive and negative clusters
    clustermemb_pos_l = (llist_li *) malloc(sizeof(llist_li));
    if (!clustermemb_pos_l) return NULL;
    llist_li_init(clustermemb_pos_l);
    clustermemb_neg_l = (llist_li *) malloc(sizeof(llist_li));
    if (!clustermemb_neg_l) return NULL;
    llist_li_init(clustermemb_neg_l);

    // compute all correlations between pivot and remaining time series
    // and create positive and negative protoclusters
    iter_li = timeseries_l.first;
    while (iter_li != NULL) {
      iter_li_next = iter_li->next; // store successor before relinking
      rho = pearson2(d, pivot, iter_li->data, l);
      corr_count++;
      if (isnan(rho)) {
        // NOTE: we add the tested time series to the noise cluster, this might not be
        // a good idea if nan value occurs because there are no overlapping valid time steps
        // in pivot and tested time series
        llist_li_relink(iter_li, &timeseries_l, noise_l);
      } else {
        if (rho >=  alpha) llist_li_relink(iter_li, &timeseries_l, clustermemb_pos_l);
        if (rho <= -alpha) llist_li_relink(iter_li, &timeseries_l, clustermemb_neg_l);
      }
      iter_li = iter_li_next;
    }

    // check whether protoclusters fulfill the minimium size constraints
    if (llist_li_size(clustermemb_pos_l) >= kappa) {
      // add to final clustering
      llist_ptr_push_back(&cluster_l, clustermemb_pos_l);
    } else {
      // relink all time series to noise cluster
      llist_li_relink_all(clustermemb_pos_l, noise_l);
      free(clustermemb_pos_l);
    }
    if (llist_li_size(clustermemb_neg_l) >= kappa) {
      // add to final clustering
      llist_ptr_push_back(&cluster_l, clustermemb_neg_l);
    } else {
      // relink all time series to noise cluster
      llist_li_relink_all(clustermemb_neg_l, noise_l);
      free(clustermemb_neg_l);
    }
  }
  printf("\rclustering finished with %ld correlation computations.\n", corr_count);

  // prepare output array with cluster assignments
  // skip noise cluster (membs id=0 during initialization)
  i = 1;
  iter_ptr = cluster_l.first->next;
  while (iter_ptr != NULL) {
    iter_li = ((llist_li *) iter_ptr->data)->first;
    while (iter_li != NULL) {
      membs[iter_li->data] = i;
      iter_li = iter_li->next;
    }
    llist_li_destroy((llist_li *) iter_ptr->data);
    free(iter_ptr->data);
    iter_ptr = iter_ptr->next;
    i++;
  }
  llist_ptr_destroy(&cluster_l);
  llist_li_destroy(&timeseries_l);

  return membs;
}

// assumes that the key really is somewhere in the array
long int binary_search_li(long int key, long int *arr, long int len) {
  long int liml, limr, rpos;
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

// COREQ
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
coreq(const double *d, long int n, long int l, double alpha, coreq_estimation_strategy_t est_strat,
    long int **membs, long int **pivots, double **cluster_corrs,
    long int *n_clus, long int *corr_comps)
{
  long int pivot, remaining, i, j, k, rpos, sample_size;
  double rho, sum_ij;
  llist_li timeseries_l; // holds all unprocessed time series
  llist_li pivot_l; // holds all pivot objects selected so far
  llist_li *clustermemb_l; // holds all time series assigned to a cluster
  llist_ptr cluster_l; // holds all clusters
  llist_ptr correlations_idx_l; // holds corrs between all pivots and time series (indices)
  llist_ptr correlations_val_l; // holds corrs between all pivots and time series (correlations)
  llist_li  correlations_cnt_l; // holds the number of entries in the previous lists
  llist_item_li *iter_li, *iter_li_next;
  llist_item_ptr *iter_ptr, *iter_idx, *iter_val;
  long int **cluster_arr; // hold all clusters with their members
  long int *cluster_size_arr; // hold all cluster sizes

  // precompute some data statistics for fast correlation computation
  double *sums = (double *) calloc(n, sizeof (double));
  double *sumsqs = (double *) calloc(n, sizeof (double));
  if ((!sums) || (!sumsqs)) return NULL;
#pragma omp parallel for
  for (i = 0; i < n; i++) {
#pragma omp simd
    for (k = 0; k < l; k++) {
      sums[i] += d[i*l+k];
      sumsqs[i] += d[i*l+k]*d[i*l+k];
    }
  }

  *membs = (long int *) calloc(n, sizeof(long int));
  if (!*membs) return NULL;

  // initialize time series index list
  llist_li_init(&timeseries_l);
  for (i = 0; i < n; i++) {
    llist_li_push_back(&timeseries_l, i);
  }

  // initialize lists
  llist_ptr_init(&cluster_l);
  llist_ptr_init(&correlations_idx_l);
  llist_ptr_init(&correlations_val_l);
  llist_li_init(&correlations_cnt_l);
  llist_li_init(&pivot_l);

  // iterate over all time series until none is left
  *corr_comps = 0;
  while (llist_li_size(&timeseries_l) > 0) {
    remaining = llist_li_size(&timeseries_l);
    printf("\r% 9ld left...", remaining);

    // select pivot time series and create its correlation container
    pivot = llist_li_front(&timeseries_l);
    llist_li_push_back(&pivot_l, pivot);
    llist_ptr_push_back(&correlations_idx_l, (long int *) calloc(remaining, sizeof (long int)));
    llist_ptr_push_back(&correlations_val_l, (double *) calloc(remaining, sizeof (double)));
    llist_li_push_back(&correlations_cnt_l, remaining);

    // initialize cluster container
    clustermemb_l = (llist_li *) malloc(sizeof (llist_li));
    if (!clustermemb_l) return NULL;
    llist_li_init(clustermemb_l);

    // compute all correlations between pivot and remaining time series
    iter_li = timeseries_l.first;
    i = 0;
    while (iter_li != NULL) {
      iter_li_next = iter_li->next; // store successor before relinking

      // compute correlation
      sum_ij = 0.0;
#pragma omp simd
      for (k = 0; k < l; k++) {
        sum_ij += d[pivot*l+k]*d[(iter_li->data)*l+k];
      }
      rho = (l*sum_ij-sums[pivot]*sums[iter_li->data])/sqrt((l*sumsqs[pivot]-sums[pivot]*sums[pivot])
              *(l*sumsqs[iter_li->data]-sums[iter_li->data]*sums[iter_li->data]));
      (*corr_comps)++;
      ((long int *) llist_ptr_back(&correlations_idx_l))[i] = iter_li->data;
      ((double *) llist_ptr_back(&correlations_val_l))[i] = rho;

      // add time series to cluster
      if (rho >= alpha) {
        llist_li_relink(iter_li, &timeseries_l, clustermemb_l);
      }

      iter_li = iter_li_next;
      i++;
    }

    // add to final clustering
    llist_ptr_push_back(&cluster_l, clustermemb_l);
  }
  *n_clus = llist_li_size(&pivot_l);
  printf("\rclustering finished with %ld correlation computations --- %ld clusters detected\n", *corr_comps, *n_clus);

  // prepare output array with cluster assignments
  // and buffer all clusters in cluster_arr for O(1) access to members
  cluster_arr = (long int **) calloc(*n_clus, sizeof (long int *));
  cluster_size_arr = (long int *) calloc(*n_clus, sizeof (long int));
  i = 0;
  iter_ptr = cluster_l.first;
  while (iter_ptr != NULL) {
    cluster_arr[i] = (long int *) calloc(llist_li_size((llist_li *) iter_ptr->data), sizeof (long int));
    cluster_size_arr[i] = llist_li_size((llist_li *) iter_ptr->data);
    j = 0;
    iter_li = ((llist_li *) iter_ptr->data)->first;
    while (iter_li != NULL) {
      cluster_arr[i][j] = iter_li->data;
      (*membs)[iter_li->data] = i;
      iter_li = iter_li->next;
      j++;
    }
    llist_li_destroy((llist_li *) iter_ptr->data);
    free(iter_ptr->data);
    iter_ptr = iter_ptr->next;
    i++;
  }

  // prepare output array with pivots
  *pivots = (long int *) calloc(*n_clus, sizeof (long int));
  i = 0;
  iter_li = pivot_l.first;
  while (iter_li != NULL) {
    (*pivots)[i] = iter_li->data;
    iter_li = iter_li->next;
    i++;
  }

  // prepare output array with correlation estimates in O(K*(K+1)/2 * log2(N) * log2(N))
  // NOTE: we use binary search to look up the precomputed pivot-time series correlations;
  // no additional correlation computations are necessary, which would require O(K*(K+1)/2 * T * log2(N))
  *cluster_corrs = (double *) calloc((*n_clus)*(*n_clus+1)/2, sizeof (double));
  iter_idx = correlations_idx_l.first;
  iter_val = correlations_val_l.first;
  iter_li = correlations_cnt_l.first;
  for (i = 0; i < (*n_clus); i++) { // loop over all pairs of pivots
    for (j = i; j < (*n_clus); j++) {
      switch (est_strat) {
        case COREQ_PIVOT:
          // search for pivot j in the correlation index list of pivot i
          rpos = binary_search_li((*pivots)[j], ((long int *) iter_idx->data), iter_li->data);
          // retrieve pivot i-j correlation from position correlation value list at position rpos
          (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] = ((double *) iter_val->data)[rpos];
          break;
        case COREQ_PIVOT_GUARANTEE:
          // same as above, but with scaling alpha-dependent scaling factor
          rpos = binary_search_li((*pivots)[j], ((long int *) iter_idx->data), iter_li->data);
          (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] = 0.5*(1.0+alpha*alpha) * ((double *) iter_val->data)[rpos];
          break;
        case COREQ_AVERAGE:
        default:
          // sample log2(N_k) precomputed correlations and use average as estimate
          sample_size = fmax(1,ceil(log2(cluster_size_arr[j])));
          (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] = 0;
          for (k = 0; k < sample_size; k++) {
            // sample random member of cluster j
            long int sample = rand() % cluster_size_arr[j];
            rpos = binary_search_li(cluster_arr[j][sample], ((long int *) iter_idx->data), iter_li->data);
            (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] += ((double *) iter_val->data)[rpos];
          }
          (*cluster_corrs)[i*(*n_clus)-i*(i+1)/2+j] /= sample_size;
      }
    }

    iter_idx = iter_idx->next;
    iter_val = iter_val->next;
    iter_li = iter_li->next;
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
  llist_li_destroy(&correlations_cnt_l);
  llist_li_destroy(&timeseries_l);
  llist_li_destroy(&pivot_l);

  free(sums);
  free(sumsqs);
  return *membs;
}

// compute aggregated losses for evaluation:
// absolute deviation, squared deviation, maximum deviation
//
// d: data set with n rows and l columns (may be NULL if corr_triu specified)
// corr_triu: precomputed true correlations in triu array (may be NULL if d specified)
// corr_clus_triu: precomputed cluster correlations in triu array
// membs: cluster membership vector of length n
// n: number of time series (rows in d)
// l: number of time steps (columns in d)
// k: number of clusters
int
compute_loss(const double *d, const double *corr_triu, const double *corr_clus_triu, const long int *membs,
      long int n, long int l, long int k,
      double *loss_abs, double *loss_sq, double *loss_max, long int *elements) {
  long int i, j, ii, jj;
  double corr_est, corr_tru;
  double loss_abs0 = 0., loss_sq0 = 0., loss_max0 = 0.;
  long int elements0 = 0;
  int abort = 0;

  if ((d == NULL) && (corr_triu == NULL)) {
    return -1;
  }

  #pragma omp parallel for private(i, j, corr_tru, corr_est, ii, jj) \
                           reduction(+:loss_abs0,loss_sq0,elements0) \
                           reduction(max:loss_max0) \
                           schedule(dynamic)
  for (i = 0; i < n; i++) {
    if ((membs[i] < 0) || (membs[i] >= k)) {
      // Invalid cluster index (must have range 0, ..., k-1). Noise cluster 0 missing?
      abort = 1;
      #pragma omp flush (abort)
    }
    for (j = i; j < n; j++) {
      // for error handling
      #pragma omp flush (abort)
      if (abort)
          continue;

      if ((membs[j] < 0) || (membs[j] >= k)) {
        // Invalid cluster index (must have range 0, ..., k-1). Noise cluster 0 missing?
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

      if (!(isnan(corr_tru) || isnan(corr_est))) {
        elements0 += 1;
        loss_abs0 += fabs(corr_tru-corr_est);
        loss_sq0  += (corr_tru-corr_est)*(corr_tru-corr_est);
        if (loss_max0 < fabs(corr_tru-corr_est)) {
            loss_max0 = fabs(corr_tru-corr_est);
        }
      }
    }
  }
  if (abort) {
      return -2;
  }

  *loss_abs = loss_abs0;
  *loss_sq = loss_sq0;
  *loss_max = loss_max0;
  *elements = elements0;
  return 0;
}

