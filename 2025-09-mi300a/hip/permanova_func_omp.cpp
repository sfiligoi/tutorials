// OpenMP version of pmn_f_stat_sW
// Implements the logic in the simplest possible way

#include <cstdlib>
#include <stdint.h>
#include <algorithm>


extern "C" int pmn_get_max_parallelism_omp() {
  // No good way to interrogate about GPU cabilities
  // Constant picked after some experimental benchamrking

  // 1k is enough for consumer-grade GPUs
  // 4k needed for larger GPUs
  // Keeping it at 4k not slowing down consumer GPUs
  // Getting much higher than 4k seems to slow down the compute
  return 4000; // should likely by dynamic, but there are no portable functions avaialble
}

template<class TFloat>
static inline void pmn_f_stat_sW_omp(
		const uint32_t n_dims,
		const TFloat * mat,
		const uint32_t n_grouping_dims,
		const uint32_t *groupings,
		const TFloat *inv_group_sizes,
		TFloat *group_sWs) {
 const uint64_t groupings_size = uint64_t(n_dims)*uint64_t(n_grouping_dims);
#pragma omp target teams distribute
 for (uint32_t grouping_el=0; grouping_el < n_grouping_dims; grouping_el++) {
    const uint32_t *grouping = groupings + uint64_t(grouping_el)*uint64_t(n_dims);
    // Use full precision for intermediate compute, to minimize accumulation errors
    double s_W = 0.0;
    for (uint32_t row=0; row < (n_dims-1); row++) {   // no columns in last row
      uint32_t group_idx = grouping[row];
#pragma omp parallel for reduction(+:s_W)
      for (uint32_t col=row+1; col < n_dims; col++) { // diagonal is always zero
        if (grouping[col] == group_idx) {
            const TFloat * mat_row = mat + uint64_t(row)*uint64_t(n_dims);
            TFloat val = mat_row[col];  // mat[row,col];
            s_W += val * val * inv_group_sizes[group_idx];
        }
      }
    }
    group_sWs[grouping_el] = s_W;
 } 
}

extern "C" void pmn_f_stat_sW_omp_fp32(
                const uint32_t n_dims,
                const float * mat,
                const uint32_t n_grouping_dims,
                const uint32_t *groupings,
                const float *inv_group_sizes,
                float *group_sWs) {
  pmn_f_stat_sW_omp<float>(n_dims,mat,n_grouping_dims,groupings,inv_group_sizes,group_sWs);
}

extern "C" void pmn_f_stat_sW_omp_fp64(
                const uint32_t n_dims,
                const double * mat,
                const uint32_t n_grouping_dims,
                const uint32_t *groupings,
                const double *inv_group_sizes,
                double *group_sWs) {
  pmn_f_stat_sW_omp<double>(n_dims,mat,n_grouping_dims,groupings,inv_group_sizes,group_sWs);
}

