#include <stdio.h>
#include <string> 
#include <algorithm>
#include <cmath>
#include <chrono>

/*
 * Initialize a buffer of n_els floats
 */
void bench_init(uint64_t n_els) {
  // Using the CPU, no threading
  {
    float* a = new float[n_els];
    auto t1 = std::chrono::steady_clock::now();
    for (uint64_t i=0; i<n_els; i++) a[i] = i*1.1;
    auto t2 = std::chrono::steady_clock::now();
    auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    // compute min just to avoid over-optimization by compiler
    printf("[%16lu] Seq CPU took %.3f s [res: %.2f]\n", n_els, time_span1.count(), *std::min_element(a,a+n_els));
    delete[] a;
  }
  // Using the CPU, but using threading
  {
    float* a = new float[n_els];
    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (uint64_t i=0; i<n_els; i++) a[i] = i*1.1;
    auto t2 = std::chrono::steady_clock::now();
    auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    // compute min just to avoid over-optimization by compiler
    printf("[%16lu] OMP CPU took %.3f s [res: %.2f]\n", n_els, time_span1.count(), *std::min_element(a,a+n_els));
    delete[] a;
  }
  // using the GPU (all SUs)
  {
    float* a = new float[n_els];
    auto t1 = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for 
    for (uint64_t i=0; i<n_els; i++) a[i] = i*1.1;
    auto t2 = std::chrono::steady_clock::now();
    auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    // compute min just to avoid over-optimization by compiler
    printf("[%16lu] Direct GPU took %.3f s [res: %.2f]\n", n_els, time_span1.count(), *std::min_element(a,a+n_els));
    delete[] a;
  }
  // Using the CPU (threaded), update buffer more than once
  {
    float* a = new float[n_els];
    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (uint64_t i=0; i<n_els; i++) a[i] = i*1.1;
    auto t2 = std::chrono::steady_clock::now();
    auto m1 = *std::min_element(a,a+n_els); // compute min just to avoid over-optimization by compiler
    auto t1b = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (uint64_t i=0; i<n_els; i++) a[i] = -1.0+i*0.1;
    auto t2b = std::chrono::steady_clock::now();
    auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    auto time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t2b - t1b);
    // compute min just to avoid over-optimization by compiler
    printf("[%16lu] Repeat OMP CPU took %.3f s + %0.3f s [res: %.2f]\n", n_els, time_span1.count(), time_span2.count(), *std::min_element(a,a+n_els)+m1);
    delete[] a;
  }
  // Using the GPU (threaded), update buffer more than once
  {
    float* a = new float[n_els];
    auto t1 = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for 
    for (uint64_t i=0; i<n_els; i++) a[i] = i*1.1;
    auto t2 = std::chrono::steady_clock::now();
    auto m1 = *std::min_element(a,a+n_els); // compute min just to avoid over-optimization by compiler
    auto t1b = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for 
    for (uint64_t i=0; i<n_els; i++) a[i] = -1.0+i*0.1;
    auto t2b = std::chrono::steady_clock::now();
    auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    auto time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t2b - t1b);
    // compute min just to avoid over-optimization by compiler
    printf("[%16lu] Repeat GPU took %.3f s + %0.3f s [res: %.2f]\n", n_els, time_span1.count(), time_span2.count(), *std::min_element(a,a+n_els)+m1);
    delete[] a;
  }
  // Initialize fist on CPU, then on GPU
  {
    float* a = new float[n_els];
    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (uint64_t i=0; i<n_els; i++) a[i] = i*1.1;
    auto t2 = std::chrono::steady_clock::now();
    auto m1 = *std::min_element(a,a+n_els); // compute min just to avoid over-optimization by compiler
    auto t1b = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for 
    for (uint64_t i=0; i<n_els; i++) a[i] = -1.0+i*0.1;
    auto t2b = std::chrono::steady_clock::now();
    auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    auto time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t2b - t1b);
    // compute min just to avoid over-optimization by compiler
    printf("[%16lu] CPU + GPU took %.3f s + %0.3f s [res: %.2f]\n", n_els, time_span1.count(), time_span2.count(), *std::min_element(a,a+n_els)+m1);
    delete[] a;
  }
  // Ping pong between CPU and GPU writing to memory
  {
    float* a = new float[n_els];
    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (uint64_t i=0; i<n_els; i++) a[i] = i*1.1;
    auto t2 = std::chrono::steady_clock::now();
    auto m1 = *std::min_element(a,a+n_els); // compute min just to avoid over-optimization by compiler
    auto t1b = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for 
    for (uint64_t i=0; i<n_els; i++) a[i] = -1.0+i*0.1;
    auto t2b = std::chrono::steady_clock::now();
    auto m2 = *std::min_element(a,a+n_els); // compute min just to avoid over-optimization by compiler
    auto t1c = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (uint64_t i=0; i<n_els; i++) a[i] = -0.2+i*0.3;
    auto t2c = std::chrono::steady_clock::now();
    auto m3 = *std::min_element(a,a+n_els); // compute min just to avoid over-optimization by compiler
    auto t1d = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for 
    for (uint64_t i=0; i<n_els; i++) a[i] = 65504.5-i*2.345;
    auto t2d = std::chrono::steady_clock::now();
    auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    auto time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t2b - t1b);
    auto time_span3 = std::chrono::duration_cast<std::chrono::duration<double>>(t2c - t1c);
    auto time_span4 = std::chrono::duration_cast<std::chrono::duration<double>>(t2d - t1d);
    // compute min just to avoid over-optimization by compiler
    auto m4 = *std::min_element(a,a+n_els); // compute min just to avoid over-optimization by compiler
    printf("[%16lu] 2x(CPU+GPU) took %.3f s + %0.3f s + %.3f s + %0.3f s [res: %.2f]\n", 
		    n_els, 
		    time_span1.count(), time_span2.count(), time_span3.count(), time_span4.count(),
		    m1+m2+std::max(m3,m4));
    delete[] a;
  }
}

int main() {
  for (int i=1; i<=2; i++) {
    printf("--- Try %i\n",i);
    bench_init(10000000l);
    bench_init(1000000000l);
  }
}
