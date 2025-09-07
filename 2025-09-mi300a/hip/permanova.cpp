#include <stdio.h>
#include <string> 
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

/*
 * Define all external functions here for convenience.
 * Production code would have had a dedicated header file.
 */

extern "C" int pmn_get_max_parallelism_omp();
extern "C" void pmn_f_stat_sW_omp_fp32(
                const uint32_t n_dims,
                const float * mat,
                const uint32_t n_grouping_dims,
                const uint32_t *groupings,
                const float *inv_group_sizes,
                float *group_sWs);

extern "C" void pmn_f_stat_sW_omp_fp64(
                const uint32_t n_dims,
                const double * mat,
                const uint32_t n_grouping_dims,
                const uint32_t *groupings,
                const double *inv_group_sizes,
                double *group_sWs);

extern "C" int pmn_get_max_parallelism_hip();
extern "C" void pmn_f_stat_sW_hip_fp32(
                const uint32_t n_dims,
                const float * mat,
                const uint32_t n_grouping_dims,
                const uint32_t *groupings,
                const float *inv_group_sizes,
                float *group_sWs);

extern "C" void pmn_f_stat_sW_hip_fp64(
                const uint32_t n_dims,
                const double * mat,
                const uint32_t n_grouping_dims,
                const uint32_t *groupings,
                const double *inv_group_sizes,
                double *group_sWs);
//
// Helper function for benchmarking
//

void bench_one(const uint32_t n_dims, const uint32_t n_grouping_dims, const char msg[]) {
  // The compute speed does not depend on mat or inv_group_sizes values, so we wil just use fake, deterministic values for benchmarking
  // It does depend on the values of grouping, using deterministic random for benchmarking
  // We also use fp64 doubles
  
  double* inv_group_sizes = new double[256]; // this one is trivially small, just keep one copy
  double* mat = new double[n_dims*n_dims];
  double* mat2 = new double[n_dims*n_dims];
  const uint64_t groupings_size = uint64_t(n_dims)*uint64_t(n_grouping_dims);
  uint32_t* groupings = new uint32_t[groupings_size];
  uint32_t* groupings2 = new uint32_t[groupings_size];
  // out buffers
  double* group_sWs = new double[n_grouping_dims];
  double* group_sWs2 = new double[n_grouping_dims];

  // not benchmarking the initialization
  std::mt19937 myRandomGenerator(1);
  // this must be serial
  for (uint32_t i=0; i<groupings_size; i++) {
     groupings[i] = myRandomGenerator()%231; // it is typical to have only a few groups
     groupings2[i] = groupings[i];
  }

  for (uint32_t i=0; i<256; i++) {
    inv_group_sizes[i] = 1.0/(2*i+3);
  }
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
  for (uint32_t row=0; row<n_dims; row++) {
     mat[n_dims*row+row] = 0.0;
     mat2[n_dims*row+row] = 0.0;
    for (uint32_t col=row+1; col<n_dims; col++) {
     mat[n_dims*row+col] = 0.5*(row%10)+sin(0.1*col);
     mat[n_dims*col+row] = mat[n_dims*row+col]; // make it symmetric
     mat2[n_dims*row+col] = mat[n_dims*row+col];
     mat2[n_dims*col+row] = mat[n_dims*row+col];
    }
  }


  for (int i=1; i<=2; i++) {
    uint32_t max_loop_step = pmn_get_max_parallelism_omp();
    auto t1 = std::chrono::high_resolution_clock::now();
    for (uint32_t dg=0; dg<n_grouping_dims; dg+=max_loop_step) {
      uint32_t loop_grouping_dims = std::min(n_grouping_dims-dg,max_loop_step); 
      pmn_f_stat_sW_omp_fp64(n_dims, mat, loop_grouping_dims, groupings+dg*uint64_t(n_dims), inv_group_sizes, group_sWs+dg);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    if(i>1) printf("[OpenMP try %i] %s (%5ix%5ix%6i): took %.3f s\n", i, msg, n_dims, n_dims, n_grouping_dims, time_span.count());
  }
  for (int i=1; i<=2; i++) {
    uint32_t max_loop_step = pmn_get_max_parallelism_hip();
    auto t1 = std::chrono::high_resolution_clock::now();
    for (uint32_t dg=0; dg<n_grouping_dims; dg+=max_loop_step) {
      uint32_t loop_grouping_dims = std::min(n_grouping_dims-dg,max_loop_step); 
      pmn_f_stat_sW_hip_fp64(n_dims, mat2, loop_grouping_dims, groupings2+dg*uint64_t(n_dims), inv_group_sizes, group_sWs2+dg);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    if(i>1) printf("[HIP    try %i] %s (%5ix%5ix%6i): took %.3f s\n", i, msg, n_dims, n_dims, n_grouping_dims, time_span.count());
  }

  delete[] group_sWs;
  delete[] group_sWs2;
  delete[] groupings2;
  delete[] groupings;
  delete[] mat2;
  delete[] mat;
  delete[] inv_group_sizes;
}

int main() {
  bench_one(2794,  1000, "Small ");
  bench_one(2794, 10000, "Small ");
  bench_one(2794,100000, "Small ");
  bench_one(8382,  1000, "Medium");
  bench_one(8382, 10000, "Medium");
  bench_one(8382,100000, "Medium");
  bench_one(25145, 1000, "Large ");
  bench_one(25145,10000, "Large ");
}
