/*  Simple mixing of serial and multi-threaded (OpenMP) loop example
 *
 *  Compile with
 *  nvc++ -mp -o mix_omp mix_omp.cpp
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
		// Make this loop parallel
#pragma omp parallel for
		for (int i=j; i<N; i++) {
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

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	delete[] C;
	delete[] B;
	delete[] A;
}

int main(int argc, char* argv[]) {
	// artificially limit, to be nice to other tutorial users
	omp_set_num_threads(10);

	compute(1000);
	compute(10000);
	compute(100000);
	compute(1000000);
	compute(10000000);
	compute(100000000);

	return 0;
}