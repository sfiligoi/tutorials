// HIP version of pmn_f_stat_sW
// Uses GPU's shared memory to improve memory access pattern

#include <cstdlib>
#include <algorithm>

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#else

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdexcept>

#endif

extern "C" int pmn_get_max_parallelism_hip() {
#ifdef USE_CUDA
  int deviceID;
  cudaDeviceProp props;

  if (cudaGetDevice(&deviceID)!=cudaSuccess) return 4000; // should never get in here, but just in case
  cudaGetDeviceProperties(&props, deviceID);

  // GPUs typically need at least 64 blocks per SM to be fully loaded
  // We want a few multiples of that to deal with unbalanced load
  // Most permanovas are multiple of 100, so double that makes a good constant
  return 200*props.multiProcessorCount;

#else
  int deviceID = 0;
  hipDeviceProp_t props;

  if (hipGetDevice(&deviceID)!=hipSuccess) return 4000; // should never get in here, but just in case
  if (hipGetDeviceProperties(&props, deviceID)!=hipSuccess) throw std::runtime_error("hipGetDeviceProperties failed");

  // GPUs typically need at least 64 blocks per SM to be fully loaded
  // We want a few multiples of that to deal with unbalanced load
  // Most permanovas are multiple of 100, so double that makes a good constant
  return 200*props.multiProcessorCount;
}
#endif

template<class TFloat>
__global__ void pmn_f_stat_sW_hip_one(
		const uint32_t n_dims,
		const TFloat * mat,
		const uint32_t n_grouping_dims,
		const uint32_t *groupings,
		const TFloat *inv_group_sizes,
		TFloat *group_sWs) {
    const uint32_t grouping_el = blockIdx.x;
    const uint32_t icol = threadIdx.x;
    const uint32_t TILE = blockDim.x;
    __shared__ double s_W_arr[128]; // we will not have more than 128 threads
    __shared__ uint32_t row_grouping[128]; // we will not have more than 128 threads

    const uint32_t *grouping = groupings + uint64_t(grouping_el)*uint64_t(n_dims);
    // Use full precision for intermediate compute, to minimize accumulation errors
    double s_W = 0.0;

    for (uint32_t trow=0; trow < (n_dims-1); trow+=TILE) {   // no columns in last row
     for (uint32_t tcol=trow+1; tcol < n_dims; tcol+=TILE) { // diagonal is always zero
      const uint32_t max_row = min(trow+TILE,n_dims-1);
      const uint32_t max_col = min(tcol+TILE,n_dims);

      __syncthreads();
      // read grouping[row] in advance into shared memory, using the whole block
      // helps with read coalesence here, and shared memory is OK to read sparse
      {
	 const uint32_t irow = threadIdx.x;
         uint32_t row=trow+irow;
	 if (row < max_row) row_grouping[irow] = grouping[row];
      }
      __syncthreads();

      TFloat local_s_W = 0.0;
      for (uint32_t row=trow; row < max_row; row++) {
        const uint32_t min_col = max(tcol,row+1);
	const uint32_t irow = row-trow;
        uint32_t group_idx = row_grouping[irow];

        const TFloat * mat_row = mat + uint64_t(row)*uint64_t(n_dims);
	uint32_t col= min_col + icol;
        if ( col < max_col) { // since TILE==col_stride, we get at most one thread picking this up
            if (grouping[col] == group_idx) {
                TFloat val = mat_row[col];  // mat[row,col];
                local_s_W += val * val * inv_group_sizes[group_idx];
            }
        }
      }
      s_W += local_s_W;
     } // for tcol
    } // for trow

    // No "reduction" instuction, must implement it ourselves
    // This is actually a very naive implementation, but it is cheap

    // now we need to collect data from all the threads
    s_W_arr[icol] = s_W;
    __syncthreads();
    // first sum across workers (4*32)
    if (threadIdx.x<32) {
      s_W = 0.0;
      for (uint32_t t=threadIdx.x; t < blockDim.x; t+=32) {
        s_W += s_W_arr[t]; 
      }
      s_W_arr[threadIdx.x] = s_W;
    }
    __syncthreads();
    // then sum using partial warps (4*8)
    if (threadIdx.x<8) {
      s_W = 0.0;
      for (uint32_t t=threadIdx.x; t < 32; t+=8) {
        s_W += s_W_arr[t]; 
      }
      s_W_arr[threadIdx.x] = s_W;
    }
    __syncthreads();
    // finally sum the remainder and write to global memory
    if (threadIdx.x==0) {
      s_W = 0.0;
      for (uint32_t t=threadIdx.x; t < 8; t++) {
        s_W += s_W_arr[t]; 
      }
      group_sWs[grouping_el] = s_W;
    }
}

template<class TFloat>
static inline void pmn_f_stat_sW_hip(
		const uint32_t n_dims,
		const TFloat * mat,
		const uint32_t n_grouping_dims,
		const uint32_t *groupings,
		const TFloat *inv_group_sizes,
		TFloat *group_sWs) {
  pmn_f_stat_sW_hip_one<TFloat><<<n_grouping_dims,128>>>(n_dims,mat,n_grouping_dims,groupings,inv_group_sizes,group_sWs);
#ifdef USE_CUDA
  if (cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("cudaDeviceSynchronize failed");
#else
  if (hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed");
#endif
}

extern "C" void pmn_f_stat_sW_hip_fp32(
                const uint32_t n_dims,
                const float * mat,
                const uint32_t n_grouping_dims,
                const uint32_t *groupings,
                const float *inv_group_sizes,
                float *group_sWs) {
  pmn_f_stat_sW_hip<float>(n_dims,mat,n_grouping_dims,groupings,inv_group_sizes,group_sWs);
}

extern "C" void pmn_f_stat_sW_hip_fp64(
                const uint32_t n_dims,
                const double * mat,
                const uint32_t n_grouping_dims,
                const uint32_t *groupings,
                const double *inv_group_sizes,
                double *group_sWs) {
  pmn_f_stat_sW_hip<double>(n_dims,mat,n_grouping_dims,groupings,inv_group_sizes,group_sWs);
}

