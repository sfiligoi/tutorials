/*  Simple logic and mock dynamic memory in loop example on GPU (OpenMP)
 *
 *  Compile with
 *  nvc++ -mp=gpu -gpu=managed -o dyn_gpu dyn_gpu.cpp
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>

void compute(const int N) {
	std::vector<float> A(N);
	std::vector<float> B(N);
	std::vector<float> C(N);
	std::vector<float> D;

	// Cannot modify size of D in GPU code.
	// Compared to the CPU version, we thus pre-allocated the maximum sizeahead of time
	// and use a separate counter
	D.resize(100);
	int D_size = 0;

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
		// this loop will be parallelized on the GPU
		// using managed memory for convenience
		int d_inc=0;
#pragma omp target teams distribute parallel for simd reduction(+:d_inc)
		for (int i=j; i<N; i++) {
			C[i-j] += A[i]*B[i];
			if ((i==j) && (C[0]>80.0)) {
				// no race condition, can be in parallel loop
				// but push_back can allocate, so compiler refuses to build
				// D.push_back(C[0]);
				// we thus simulate it
				D[D_size] = C[0];
				d_inc += 1;
			} else {
				A[i] += 1.e-9*C[i-j];
				B[i] -= 1.e-10*C[i-j];
			}
		}
		D_size += d_inc;
	}
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] C0 = %f (%i) (took %.3f s)\n", N,C[0], int(D_size), time_span.count());
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
