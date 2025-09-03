#include <stdio.h>
#include <string> 
#include <algorithm>
#include <cmath>
#include <chrono>

// Simplified version of OMP-AWARE STREAM benchmark
// https://github.com/sfiligoi/STREAM-OMPGPU/blob/master/stream.c
template<typename TReal>
void triad(uint64_t STREAM_ARRAY_SIZE, TReal scalar, TReal * __restrict__ a, TReal * __restrict__ b,TReal * __restrict__ c) {
#ifdef OMPGPU
#pragma omp target teams distribute parallel for 
#else
#pragma omp parallel for
#endif
  for (uint64_t j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = b[j]+scalar*c[j];
  }
}

void bench_triad(int try_num, uint64_t STREAM_ARRAY_SIZE) {
  float s = 0.0;
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("--- Try %i, els: %lu\n", try_num, STREAM_ARRAY_SIZE);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    s = time_span1.count(); // reuse as a pseudo-number generator
  }
  {
    float* a = new float[STREAM_ARRAY_SIZE];
    float* b = new float[STREAM_ARRAY_SIZE];
    float* c = new float[STREAM_ARRAY_SIZE];
    // don't benchmark initialization
#ifdef OMPGPU
#pragma omp target teams distribute parallel for 
#else
#pragma omp parallel for
#endif
    for (uint64_t j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = 0.0;
	    b[j] = s*(j%9999)+(j%231);
	    c[j] = s*(j%7419)+(j%123);
    }
    for (int i=0; i<4; i++) {
      auto t1 = std::chrono::high_resolution_clock::now();
      triad(STREAM_ARRAY_SIZE, s, a, b, c);
      auto t2 = std::chrono::high_resolution_clock::now();
      auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
      printf("[fp32] took %.3f s\n", time_span1.count());
    }
    delete[] c;
    delete[] b;
    delete[] a;
  }
  {
    double* a = new double[STREAM_ARRAY_SIZE];
    double* b = new double[STREAM_ARRAY_SIZE];
    double* c = new double[STREAM_ARRAY_SIZE];
    // don't benchmark initialization
#ifdef OMPGPU
#pragma omp target teams distribute parallel for 
#else
#pragma omp parallel for
#endif
    for (uint64_t j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = 0.0;
	    b[j] = s*(j%9999)+(j%231);
	    c[j] = s*(j%7419)+(j%123);
    }
    for (int i=0; i<4; i++) {
      auto t1 = std::chrono::high_resolution_clock::now();
      triad<double>(STREAM_ARRAY_SIZE, s, a, b, c);
       auto t2 = std::chrono::high_resolution_clock::now();
      auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
      printf("[fp64] took %.3f s\n", time_span1.count());
    }
    delete[] c;
    delete[] b;
    delete[] a;
  }
}


int main() {
  bench_triad(1,2000000000l);
  bench_triad(2,2000000000l);
}
