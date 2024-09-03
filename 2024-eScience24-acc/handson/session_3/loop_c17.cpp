/*  Simple fixed-size vector in C++17 loop example - works both in multi-core and GPU mode, depending on compiler options
 *
 *  Compile with
 *  nvc++ --c++17 -Mvect -stdpar=multicore -o loop_c17 loop_c17.cpp
 *  or
 *  nvc++ --c++17 -stdpar=gpu -o loop_c17 loop_c17.cpp
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <chrono>
#include <algorithm>
#include <execution>

void compute(const int N) {
	std::vector<int> I(N);
	float* A = new float[N];
	float* B = new float[N];
	float* C = new float[N];

	auto t1 = std::chrono::high_resolution_clock::now();
	for (int i=0; i<N; i++) {
		int r = rand();
		A[i] = 0.5+0.01*(r%100);
		B[i] = 1.5-0.001*(r%1000);
		C[i] = 0.0;
		I[i] = i;
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will be parallelized
		// std::execution::par_unseq tells compiler it can aggressively parallelize/vectorize
		std::for_each(std::execution::par_unseq,
			I.data()+j, I.data()+100, [=] (const int i) {
			C[i-j] += A[i]*B[i];
			A[i] += 1.e-9*C[i-j];
			B[i] -= 1.e-10*C[i-j];
		});
	}
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] C0 = %f (took %.4f s)\n", N,C[0], time_span.count());
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
