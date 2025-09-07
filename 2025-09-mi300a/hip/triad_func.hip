// Implement the simple triad operation
// This file contains both HIP and C++/OpenMP code
// use -DOMPGPU for the later

#include <stdint.h>
#include <stdio.h>

#ifdef OMPGPU
// Simplified version of OMP-AWARE STREAM benchmark
// https://github.com/sfiligoi/STREAM-OMPGPU/blob/master/stream.c
template<typename TReal>
void triad(uint64_t STREAM_ARRAY_SIZE, TReal scalar, TReal * __restrict__ a, TReal * __restrict__ b,TReal * __restrict__ c) {
#pragma omp target teams distribute parallel for 
  for (uint64_t j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = b[j]+scalar*c[j];
  }
}
#else
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// Helper function to size the grid
int get_blocks() {
  int deviceID = 0;
  hipDeviceProp_t props;

  if (hipGetDevice(&deviceID)!=hipSuccess) return 1; // should never get in here, but just in case
  if (hipGetDeviceProperties(&props, deviceID)!=hipSuccess) return 1; // should never get in here, but just in case

  // GPUs typically need at least 64 blocks per SM to be fully loaded
  // We want a few multiples of that to deal with unbalanced load
  // Easier if we keep multipels of 100, so double that makes a good constant
  return 200*props.multiProcessorCount;
}

#ifndef HIP_OPT

// basic HIP function
template<typename TReal>
__global__ void triad_one(uint64_t STREAM_ARRAY_SIZE, uint32_t vector_size,
                          TReal scalar, TReal * __restrict__ a, TReal * __restrict__ b,TReal * __restrict__ c) {
  uint64_t j = blockIdx.x*vector_size+threadIdx.x;
  if (j<STREAM_ARRAY_SIZE) {// just to simplify edge case handling
      a[j] = b[j]+scalar*c[j];
  }
}

template<typename TReal>
void triad(uint64_t STREAM_ARRAY_SIZE, TReal scalar, TReal * __restrict__ a, TReal * __restrict__ b,TReal * __restrict__ c) {
  const uint32_t vector_size = 256; // Same as default OMP for AMD GPUs
  const uint32_t n_blocks = (STREAM_ARRAY_SIZE + (vector_size-1)) / vector_size; // round up
  //printf("[INFO] Grid: %u x %u\n", n_blocks, vector_size);
  triad_one<TReal><<<n_blocks, vector_size>>>(STREAM_ARRAY_SIZE,vector_size,scalar,a,b,c);
  if (hipDeviceSynchronize()!=hipSuccess) {printf("[ERROR] hipDeviceSynchronize failed!\n");}
}


#else

// Do more work in each HIP function
template<typename TReal>
__global__ void triad_many(uint64_t STREAM_ARRAY_SIZE, uint32_t n_iters, uint32_t vector_size,
                           TReal scalar, TReal * __restrict__ a, TReal * __restrict__ b,TReal * __restrict__ c) {
  const uint64_t itr_vect_size = uint64_t(vector_size)*n_iters; // number of element processed by this vector compute
  uint64_t j = blockIdx.x*itr_vect_size+threadIdx.x;
  // This is one thread in a warp (aka wavefront)
  // All threads run in parallel, so jump vector_size
  for (uint32_t i=0; i<n_iters; i++) {
    j+=vector_size;
    if (j<STREAM_ARRAY_SIZE) {// just to simplify edge case handling
      a[j] = b[j]+scalar*c[j];
    }
  }
}

template<typename TReal>
void triad(uint64_t STREAM_ARRAY_SIZE, TReal scalar, TReal * __restrict__ a, TReal * __restrict__ b,TReal * __restrict__ c) {
  const uint32_t vector_size = 1024; // This problem preferrs slightly larger vector_size
  const uint32_t n_blocks = get_blocks();
  const uint32_t div = n_blocks*vector_size;
  const uint32_t n_iters = (STREAM_ARRAY_SIZE + (div-1)) / div; // round up
  //printf("[INFO] Grid: %u x %u, div: %u iter: %u\n", n_blocks, vector_size, div, n_iters);
  triad_many<TReal><<<n_blocks, vector_size>>>(STREAM_ARRAY_SIZE,n_iters,vector_size,scalar,a,b,c);
  if (hipDeviceSynchronize()!=hipSuccess) {printf("[ERROR] hipDeviceSynchronize failed!\n");}
}
#endif

// end HIP
#endif

extern "C" void triad64(uint64_t STREAM_ARRAY_SIZE, double scalar, double * a, double * b, double * c) {
  triad<double>(STREAM_ARRAY_SIZE,scalar,a,b,c);
}

extern "C" void triad32(uint64_t STREAM_ARRAY_SIZE, float  scalar, float  * a, float  * b, float * c) {
  triad<float>(STREAM_ARRAY_SIZE,scalar,a,b,c);
}

