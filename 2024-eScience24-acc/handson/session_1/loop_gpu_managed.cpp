/*  Simple GPU (OpenMP) loop example - Requires managed memory
 *
 *  Compile with
 *  nvc++ -mp=gpu -gpu=managed -o loop_gpu_managed loop_gpu_managed.cpp
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

void compute(const int N) {
	float* A = new float[N];
	float* B = new float[N];
	float* C = new float[N];

	auto t1 = std::chrono::high_resolution_clock::now();
	for (int i=0; i<N; i++) {
		int r = rand();
		A[i] = 0.5+0.01*(r%100);
		B[i] = 1.5-0.001*(r%1000);
		C[i] = 0.0;
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
		// assuming unified memory handling (aka NVIDIA managed memory)
#pragma omp target teams distribute parallel for simd 
		for (int i=j; i<N; i++) {
			C[i-j] += A[i]*B[i];
			A[i] += 1.e-9*C[i-j];
			B[i] -= 1.e-10*C[i-j];
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	delete[] C;
	delete[] B;
	delete[] A;
}

int main(int argc, char* argv[]) {
	compute(1000);
	compute(10000);
	compute(100000);
	compute(1000000);
	compute(10000000);
	compute(100000000);

	return 0;
}
