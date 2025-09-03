#include <stdio.h>
#include <string> 
#include <algorithm>
#include <cmath>
#include <chrono>

// C++ rough equivalent of 
// https://github.com/sfiligoi/scikit-bio/blob/b0ed7ed183ed9d128fcae8b51288e3e57ddd65a1/skbio/stats/ordination/_utils.py#L201
// Already fused several intermediate matrices
template<class TReal>
TReal* e_matrix_naive(const uint32_t n_dims, const TReal distance_matrix[]) {
    TReal* e_out = new TReal[n_dims*n_dims];
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
    for (uint32_t i=0; i<(n_dims*n_dims); i++) e_out[i] = distance_matrix[i]*distance_matrix[i] /-2;
    return e_out;
}

template<class TReal>
TReal* f_matrix_naive(const uint32_t n_dims, const TReal E_matrix[]) {
  TReal* row_means = new TReal[n_dims];
  TReal* col_means = new TReal[n_dims];
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
  for (uint32_t row=0; row<n_dims; row++) {
    double row_mean = 0.0;
    for (uint32_t col=0; col<n_dims; col++) {
      row_mean += E_matrix[n_dims*row + col];
    }
    row_means[row] = row_mean/n_dims;
  }
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
  for (uint32_t col=0; col<n_dims; col++) {
    double col_mean = 0.0;
    for (uint32_t row=0; row<n_dims; row++) {
      col_mean += E_matrix[n_dims*row + col];
    }
    col_means[col] = col_mean/n_dims;
  }

  double matrix_mean = 0.0;
#ifdef OMPGPU
#pragma omp target teams distribute parallel for reduction(+: matrix_mean)
#else
#pragma omp parallel for reduction(+: matrix_mean)
#endif
  for (uint32_t i=0; i<(n_dims*n_dims); i++) matrix_mean += E_matrix[i];
  matrix_mean = (matrix_mean/n_dims)/n_dims;

  TReal* f_out = new TReal[n_dims*n_dims];
#ifdef OMPGPU
  // collapse to get vectorization of the inner loop
#pragma omp target teams loop collapse(2)
#else
  // inner loop automatically vectorized
#pragma omp parallel for
#endif
  for (uint32_t row=0; row<n_dims; row++) {
    for (uint32_t col=0; col<n_dims; col++) {
      f_out[n_dims*row + col] = E_matrix[n_dims*row + col] - row_means[row] - col_means[col] + matrix_mean;
    }
  }
  delete[] col_means;
  delete[] row_means;

  return f_out;
}

template<class TReal>
TReal* center_distance_matrix_naive(const uint32_t n_dims, const TReal distance_matrix[]) {
    TReal* e_matrix = e_matrix_naive(n_dims, distance_matrix);
    TReal* c_matrix = f_matrix_naive(n_dims, e_matrix);
    delete[] e_matrix;
    return c_matrix;
}


// similar to the above, but avoiding any malloc

template<class TReal>
void e_matrix_nomalloc(const uint32_t n_dims, const TReal* __restrict__ distance_matrix, TReal* __restrict__ e_out) {
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
    for (uint32_t i=0; i<(n_dims*n_dims); i++) e_out[i] = distance_matrix[i]*distance_matrix[i] /-2;
}

template<class TReal>
void f_matrix_inplace_nomalloc(const uint32_t n_dims, TReal* __restrict__ mat, TReal* __restrict__ workspace) {
  // row_means and col_means are the same, so compute just one
  TReal* __restrict__ means = workspace;
  double matrix_sum = 0.0;
#ifdef OMPGPU
  // vectorized on outer loop, keep memory access contiguous by
  // having outer loop col
#pragma omp target teams distribute parallel for reduction(+: matrix_sum)
  for (uint32_t col=0; col<n_dims; col++) {
    double col_sum = 0.0;
    for (uint32_t row=0; row<n_dims; row++) {
      col_sum += mat[n_dims*row + col];
    }
    means[col] = col_sum/n_dims;
    matrix_sum += col_sum;
  }
#else
  // CPU compilers usually prefer vectorizing the inner loop
  // so keep col there to improve memory access
#pragma omp parallel for reduction(+: matrix_sum)
  for (uint32_t row=0; row<n_dims; row++) {
    double row_sum = 0.0;
    for (uint32_t col=0; col<n_dims; col++) {
      row_sum += mat[n_dims*row + col];
    }
    means[row] = row_sum/n_dims;
    matrix_sum += row_sum;
  }
#endif

  double matrix_mean = (matrix_sum/n_dims)/n_dims;

#ifdef OMPGPU
  // collapse to get vectorization of the inner loop
#pragma omp target teams loop collapse(2)
#else
  // inner loop automatically vectorized
#pragma omp parallel for
#endif
  for (uint32_t row=0; row<n_dims; row++) {
    for (uint32_t col=0; col<n_dims; col++) {
      mat[n_dims*row + col] += matrix_mean - means[row] - means[col];
    }
  }

}

// Center the matrix
// mat and center must be nxn and symmetric
// centered must be pre-allocated and same size as mat...will work even if centered==mat
// workspace must be pre-allocated
template<class TReal>
static inline void center_distance_matrix_nomalloc(const uint32_t n_dims, const TReal mat[], TReal centered[], TReal workspace[]) {
   e_matrix_nomalloc(n_dims, mat, centered);
   f_matrix_inplace_nomalloc(n_dims, centered, workspace);
}


// Variant of the center_matrix implemenation in scikit-bio-binaries:
// https://github.com/scikit-bio/scikit-bio-binaries/blob/27dfa5a357a3eef3934d8dfd7f9a2e2e217fccc1/src/ordination/principal_coordinate_analysis.cpp#L33

// one-pass e_matrix and mean values
template<class TReal>
static inline void E_matrix_means(const uint32_t n_dims,                                // IN
                           const TReal mat[],                                           // IN
                           TReal centered[], TReal row_means[], TReal &global_mean) {   // OUT
  double global_sum = 0.00;

#ifdef OMPGPU
  // use only coarse-grained parallelism on the outer loop
#pragma omp target teams loop reduction(+: global_sum)
#else
  // body automatically vectorized
#pragma omp parallel for reduction(+: global_sum)
#endif
  for (uint32_t row=0; row<n_dims; row++) {
    double row_sum = 0.0;

#ifdef OMPGPU
// proper vectorization happens here
#pragma omp parallel for reduction(+: row_sum)
#else
  // no need to help the compiler in the CPU case, it will vectorize by itself
#endif
    for (uint32_t col=0; col<n_dims; col++) {
       TReal el0 = mat[n_dims*row + col];
       el0 =  el0*el0/-2;
       centered[n_dims*row + col] = el0;
       row_sum += el0;
    }

    global_sum += row_sum;
    row_means[row] = row_sum/n_dims;
  }

  global_mean = (global_sum/n_dims)/n_dims;
}

template<class TReal>
static inline void F_matrix_inplace(const TReal * __restrict__ row_means, const TReal global_mean, TReal * __restrict__ centered, const uint32_t n_dims) {
  // we know matrix is symmetric, so exploit row_meads == col_means
#ifdef OMPGPU
  // collapse to get vectorization of the inner loop
#pragma omp target teams distribute parallel for collapse(2)
#else
  // inner loop automatically vectorized
#pragma omp parallel for
#endif
  for (uint32_t row=0; row<n_dims; row++) {
    for (uint32_t col=0; col<n_dims; col++) {
      centered[n_dims*row+col] += global_mean - row_means[row] - row_means[col];
    }
  }
}

// Center the matrix
// mat and center must be nxn and symmetric
// centered must be pre-allocated and same size as mat...will work even if centered==mat
// workspace must be pre-allocated
template<class TReal>
static inline void mat_to_centered(const uint32_t n_dims, const TReal mat[], TReal centered[], TReal workspace[]) {

   TReal global_mean;
   TReal *row_means = workspace;
   E_matrix_means(n_dims, mat, centered, row_means, global_mean);
   F_matrix_inplace(row_means, global_mean, centered, n_dims);
}

//
// Helper function for benchmarking
//

void bench_one(const uint32_t n_dims, const char msg[]) {
  // The compute speed does not depend on valus, so we wil just use fake, deterministic values for benchmarking
  // We also use fp64 doubles
  
#ifdef OMPGPU
  const char cpugpu[] = "GPU";
#else
  const char cpugpu[] = "CPU";
#endif

  double* mat = new double[n_dims*n_dims];
  double* mat2 = new double[n_dims*n_dims];
  double* mat3 = new double[n_dims*n_dims];

  // not benchmarking the initialization
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
  for (uint32_t row=0; row<n_dims; row++) {
     mat[n_dims*row+row] = 0.0;
     mat2[n_dims*row+row] = 0.0;
     mat3[n_dims*row+row] = 0.0;
    for (uint32_t col=row+1; col<n_dims; col++) {
     mat[n_dims*row+col] = 0.5*(row%10)+sin(0.1*col);
     mat[n_dims*col+row] = mat[n_dims*row+col]; // make it symmetric
     mat2[n_dims*row+col] = mat[n_dims*row+col];
     mat2[n_dims*col+row] = mat[n_dims*row+col];
     mat3[n_dims*row+col] = mat[n_dims*row+col];
     mat3[n_dims*col+row] = mat[n_dims*row+col];
    }
  }


  for (int i=1; i<=2; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    double * cent = center_distance_matrix_naive(n_dims, mat);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    // note:  res printed just to avoid over-optimization
    double res = 0.0;
#ifdef OMPGPU
#pragma omp target teams distribute parallel for reduction(+:res)
#else
#pragma omp parallel for reduction(+:res)
#endif
    for (uint32_t i=0; i<(n_dims*n_dims); i++) res+=cent[i];

    printf("[Naive    try %i] %s (%5ix%5i) on %s: took %.3f s [res: %.2f]\n", i, msg, n_dims, n_dims, cpugpu, time_span1.count(), res);
    delete[] cent; // cent buffer created in the function, so I cannot reuse it
  }

  {
    double* cent = new double[n_dims*n_dims];
    double *row_means = new double[n_dims];
    for (int i=1; i<=2; i++) {
      auto t1 = std::chrono::high_resolution_clock::now();
      center_distance_matrix_nomalloc(n_dims, mat2,cent,row_means);
      auto t2 = std::chrono::high_resolution_clock::now();
      auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

      // note:  res printed just to avoid over-optimization
      double res = 0.0;
#ifdef OMPGPU
#pragma omp target teams distribute parallel for reduction(+:res)
#else
#pragma omp parallel for reduction(+:res)
#endif
      for (uint32_t i=0; i<(n_dims*n_dims); i++) res+=cent[i];

      printf("[Nomalloc try %i] %s (%5ix%5i) on %s: took %.3f s [res: %.2f]\n", i, msg, n_dims, n_dims, cpugpu, time_span1.count(), res);
    }
    // we had reused the buffers between retries
    delete[] row_means;
    delete[] cent;
  }

  {
    double* cent = new double[n_dims*n_dims];
    double *row_means = new double[n_dims];
    for (int i=1; i<=2; i++) {
      auto t1 = std::chrono::high_resolution_clock::now();
      mat_to_centered(n_dims, mat3,cent,row_means);
      auto t2 = std::chrono::high_resolution_clock::now();
      auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

      // note:  res printed just to avoid over-optimization
      double res = 0.0;
#ifdef OMPGPU
#pragma omp target teams distribute parallel for reduction(+:res)
#else
#pragma omp parallel for reduction(+:res)
#endif
      for (uint32_t i=0; i<(n_dims*n_dims); i++) res+=cent[i];

      printf("[Tuned    try %i] %s (%5ix%5i) on %s: took %.3f s [res: %.2f]\n", i, msg, n_dims, n_dims, cpugpu, time_span1.count(), res);
    }
    // we had reused the buffers between retries
    delete[] row_means;
    delete[] cent;
  }

  delete[] mat3;
  delete[] mat2;
  delete[] mat;
}

int main() {
  bench_one(2794,  "Small ");
  bench_one(8382,  "Medium");
  bench_one(25145, "Large ");
}
