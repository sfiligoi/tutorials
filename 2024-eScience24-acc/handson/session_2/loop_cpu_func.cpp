/*  Simple multi-threaded (OpenMP) loop example with function calling
 *
 *  Compile with
 *  nvc++ -Mvect -mp -c -o loop_cpu_func_ext.o loop_cpu_func_ext.cpp
 *  nvc++ -Mvect -mp -o loop_cpu_func loop_cpu_func.cpp loop_cpu_func_ext.o
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>

void oneI(float& c, float& a, float& b) {
	c += a*b;
	a += 1.e-9*c;
	b -= 1.e-10*c;
}

// just declare here to avoid using a header file
void oneE(float& c, float& a, float& b);

void hundredI(float* C, float* A, float* B, int N) {
	for (int i=0; i<N; i++) oneI(C[i],A[i],B[i]);
}

// just declare here to avoid using a header file
void hundredE(float* C, float* A, float* B, int N);

void compute(const int N) {
	float* oA = new float[N];
	float* oB = new float[N];
	float* oC = new float[N];
	float* A = new float[N];
	float* B = new float[N];
	float* C = new float[N];

	// compute initial values once, since they are random
	for (int i=0; i<N; i++) {
		int r = rand();
		oA[i] = 0.5+0.01*(r%100);
		oB[i] = 1.5-0.001*(r%1000);
		oC[i] = 0.0;
	}


	{ // inline function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run in parallel
#pragma omp parallel for
		for (int i=j; i<N; i++) {
			oneI(C[i-j], A[i], B[i]);
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] inline C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // external function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run in parallel
#pragma omp parallel for
		for (int i=j; i<N; i++) {
			oneE(C[i-j], A[i], B[i]);
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] ext C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // inline block function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run in parallel
#pragma omp parallel for
		for (int i=j; i<N; i+=100) {
			hundredI(&(C[i-j]), &(A[i]), &(B[i]), std::min(N-i,100));
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] block C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // external block function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run in parallel
#pragma omp parallel for
		for (int i=j; i<N; i+=100) {
			hundredE(&(C[i-j]), &(A[i]), &(B[i]), std::min(N-i,100));
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] ext block C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	delete[] C;
	delete[] B;
	delete[] A;
	delete[] oC;
	delete[] oB;
	delete[] oA;
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
