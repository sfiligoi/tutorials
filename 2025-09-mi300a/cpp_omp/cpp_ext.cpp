//  Simple OpenMP loop example with function calling

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>

// Header-only external functions and be complex
// but they can still all be inlined by the compiler
#include "cpp_ext_inlines.hpp"

// Just declare here to avoid using a header file
// implementation in cpp_ext_func.cpp
// In real libraries there would have been a header file for these
void oneE(float& c, float& a, float& b);
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

   // repeat twice, to showcase initialization overhead
   for (int li=1; li<=2; li++) {

	{ // using a macro, equivalent to cut-and-paste
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run in parallel
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
		for (int i=j; i<N; i++) {
			oneM(C[i-j], A[i], B[i]);
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	// Printing C[0] to avoid over-optimization by the compiler
	printf("[%10i try %i] macro     C0 = %8.3f (took %.3f s)\n", N,li, C[0], time_span.count());
	}

	{ // inline function
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run in parallel
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
		for (int i=j; i<N; i++) {
			oneI(C[i-j], A[i], B[i]);
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	// Printing C[0] to avoid over-optimization by the compiler
	printf("[%10i try %i] inline    C0 = %8.3f (took %.3f s)\n", N,li, C[0], time_span.count());
	}

	{ // external function
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run in parallel
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
		for (int i=j; i<N; i++) {
			oneE(C[i-j], A[i], B[i]);
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	// Printing C[0] to avoid over-optimization by the compiler
	printf("[%10i try %i] ext       C0 = %8.3f (took %.3f s)\n", N,li, C[0], time_span.count());
	}

	{ // inline block function
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run in parallel
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
		for (int i=j; i<N; i+=100) {
			hundredI(&(C[i-j]), &(A[i]), &(B[i]), std::min(N-i,100));
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	// Printing C[0] to avoid over-optimization by the compiler
	printf("[%10i try %i] block     C0 = %8.3f (took %.3f s)\n", N,li, C[0], time_span.count());
	}

	{ // external block function
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run in parallel
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
		for (int i=j; i<N; i+=100) {
			hundredE(&(C[i-j]), &(A[i]), &(B[i]), std::min(N-i,100));
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	// Printing C[0] to avoid over-optimization by the compiler
	printf("[%10i try %i] ext block C0 = %8.3f (took %.3f s)\n", N,li, C[0], time_span.count());
	}
   }

   delete[] C;
   delete[] B;
   delete[] A;
   delete[] oC;
   delete[] oB;
   delete[] oA;
}

int main(int argc, char* argv[]) {
   compute(1000000);
   compute(100000000);
   return 0;
}
