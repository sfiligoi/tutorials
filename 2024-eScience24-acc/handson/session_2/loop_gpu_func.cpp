/*  Simple GPU (OpenMP) loop example with function calling
 *
 *  Compile with
 *  nvc++ -mp=gpu -c -o loop_gpu_func_ext.o loop_gpu_func_ext.cpp
 *  nvc++ -mp=gpu -o loop_gpu_func loop_gpu_func.cpp loop_gpu_func_ext.o
 */

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

/*
 * We use a simple block loop as an example of a not-trivial logic
 *    one would want to abstract away into a function.
 * Real-life code would likely be significantly more complex.
 *
 */
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
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
#pragma omp target teams distribute parallel for simd 
		for (int i=j; i<N; i++) {
			oneI(C[i-j], A[i], B[i]);
		}
	}
	// we only need C[0] on the CPU
#pragma omp target update from(C[0:1])
	// we can now delete all the other gpu buffers
#pragma omp target exit data map(delete:A[0:N],B[0:N],C[0:N])
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] inline C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // external function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
#pragma omp target teams distribute parallel for simd 
		for (int i=j; i<N; i++) {
			oneE(C[i-j], A[i], B[i]);
		}
	}
	// we only need C[0] on the CPU
#pragma omp target update from(C[0:1])
	// we can now delete all the other gpu buffers
#pragma omp target exit data map(delete:A[0:N],B[0:N],C[0:N])
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] ext C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // inline block function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
#pragma omp target teams distribute parallel for simd 
		for (int i=j; i<N; i+=100) {
			hundredI(&(C[i-j]), &(A[i]), &(B[i]), std::min(N-i,100));
		}
	}
	// we only need C[0] on the CPU
#pragma omp target update from(C[0:1])
	// we can now delete all the other gpu buffers
#pragma omp target exit data map(delete:A[0:N],B[0:N],C[0:N])
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] block C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // external block function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
#pragma omp target teams distribute parallel for simd 
		for (int i=j; i<N; i+=100) {
			hundredE(&(C[i-j]), &(A[i]), &(B[i]), std::min(N-i,100));
		}
	}
	// we only need C[0] on the CPU
#pragma omp target update from(C[0:1])
	// we can now delete all the other gpu buffers
#pragma omp target exit data map(delete:A[0:N],B[0:N],C[0:N])
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
	compute(1000);
	compute(10000);
	compute(100000);
	compute(1000000);
	compute(10000000);
	compute(100000000);

	return 0;
}
