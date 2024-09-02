/*  Simple fixed-size vector in C++ loop example (OpenMP)
 *
 *  Compile with
 *  nvc++ -Mvect -mp -o vector_cpu vector_cpu.cpp
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <chrono>

void compute(const int N) {
	std::vector<float> A(N);
	std::vector<float> B(N);
	std::vector<float> C(N);

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
		// this loop will be parallelized
#pragma omp parallel for
		for (int i=j; i<N; i++) {
			C[i-j] += A[i]*B[i];
			A[i] += 1.e-9*C[i-j];
			B[i] -= 1.e-10*C[i-j];
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
}

std::array<float,10000000> gA;
std::array<float,10000000> gB;
std::array<float,10000000> gC;

template<size_t N>
void compute2() {
	// just using a reference to keep the code identical to the above
	// but we want to use global memory
	std::array<float,N> &A = gA;
	std::array<float,N> &B = gB;
	std::array<float,N> &C = gC;

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
		// this loop will be parallelized
#pragma omp parallel for
		for (int i=j; i<N; i++) {
			C[i-j] += A[i]*B[i];
			A[i] += 1.e-9*C[i-j];
			B[i] -= 1.e-10*C[i-j];
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
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
	compute2<10000000>();

	return 0;
}
