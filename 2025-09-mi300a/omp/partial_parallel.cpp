/*  Simple mixing of serial and GPU (OpenMP) loop example
 *  Requires managed memory
 *
 *  Compile with
 *  nvc++ -mp=gpu -gpu=managed -o mix_gpu_managed mix_gpu_managed.cpp
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

void compute(const uint64_t N) {
  float* A = new float[N];
  float* B = new float[N];
  float* C = new float[N];
  for (int l=1; l<=2; l++) {
	auto t1 = std::chrono::high_resolution_clock::now();
	for (uint64_t i=0; i<N; i++) {
		int r = rand();
		A[i] = 0.5+0.01*(r%100);
		B[i] = 1.5-0.001*(r%1000);
		C[i] = 0.0;
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<500; j++) {
		// this loop will run in parallel
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
		for (uint64_t i=j; i<N; i++) {
			C[i-j] += A[i]*B[i];
			A[i] += 1.e-9*C[i-j];
			B[i] -= 1.e-10*C[i-j];
		}
		// this must be serial again, due to dependencies
		for (int i=0; i<1000; i++) {
			int r = rand();
			C[0] += 1.e-9 * C[r%N];
			C[r%N] -= 1.e-8 * C[0];
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span_init = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%10lu try %i] C0 = %f (took %.3f s init %.3f s)\n", N,l, C[0], time_span.count(), time_span_init.count());
  }
  delete[] C;
  delete[] B;
  delete[] A;
}

int main(int argc, char* argv[]) {
	compute(1000);
	compute(1000000);
	compute(100000000l);

	return 0;
}
