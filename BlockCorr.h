#ifndef BLOCKCORR_H
#define BLOCKCORR_H

typedef unsigned long coreq_estimation_strategy_t;
#define COREQ_PIVOT 0
#define COREQ_PIVOT_GUARANTEE 1
#define COREQ_AVERAGE 2

// computes correlation between rows i and j in matrix d (row-major order)
double pearson2(const double *d, const unsigned long i, const unsigned long j, const unsigned long l);

// computes size n*n vector with correlation matrix in row-major order
double *pearson(const double *d, unsigned long n, unsigned long l);

// computes size n*(n+1)/2 vector with upper triangular correlation matrix (row-major order)
double *pearson_triu(const double *d, unsigned long n, unsigned long l);

// computes size n vector with cluster assignments
unsigned long *cluster(const double *d, unsigned long n, unsigned long l, double alpha, unsigned long kappa, unsigned long max_nan);

// computes cluster assignments, pivots and inter-cluster correlations;
// also returns the number of clusters and correlation computations
long int *
coreqPP(const double *d, unsigned long n, unsigned long l, double alpha, coreq_estimation_strategy_t est_strat,
    long int **membs, long int **pivots, double **cluster_corrs,
    long int *n_clus, long int *corr_comps);


int
compute_loss(const double *d, const double *corr_triu, const double *corr_clus_triu, const long *membs,
      unsigned long n, unsigned long l, unsigned long k,
      double *loss_abs, double *loss_sq, double *loss_max, unsigned long *elements);

#endif
