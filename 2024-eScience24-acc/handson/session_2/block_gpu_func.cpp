/*  Block-based GPU (OpenMP) loop example with function calling
 *  Note: The simple block is just n example of function encapsulation
 *        and real-world problems would be significantly more complex.
 *
 *  Compile with
 *  nvc++ -mp=gpu -c -o block_gpu_func_ext.o block_gpu_func_ext.cpp
 *  nvc++ -mp=gpu -o block_gpu_func block_gpu_func.cpp block_gpu_func_ext.o
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

/*
 * We use a simple block loop as an example of a not-trivial logic
 *    one would want to abstract away into a function.
 * Real-life code would likely be significantly more complex.
 *
 */
void hundredNaive(float* C, float* A, float* B, int N) {
	for (int i=0; i<N; i++) oneI(C[i],A[i],B[i]);
}

void hundredI(float* C, float* A, float* B, int N) {
#pragma omp loop bind(parallel)
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


	{ // basic, no blocks
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
	printf("[%i] plain C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // naive block in a loop
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
		// Split outer loop among GPU compute units
#pragma omp target teams distribute parallel for simd 
		for (int i=j; i<N; i+=100) {
			auto pC = &(C[i-j]);
			auto pA = &(A[i]);
			auto pB = &(B[i]);
			auto M = std::min(N-i,100);
			for (int i=0; i<M; i++) oneI(pC[i],pA[i],pB[i]);
		}
	}
	// we only need C[0] on the CPU
#pragma omp target update from(C[0:1])
	// we can now delete all the other gpu buffers
#pragma omp target exit data map(delete:A[0:N],B[0:N],C[0:N])
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] naive block C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // naive block invoked as a function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
		// Split outer loop among GPU compute units
#pragma omp target teams distribute parallel for simd 
		for (int i=j; i<N; i+=100) {
			hundredNaive(&(C[i-j]), &(A[i]), &(B[i]), std::min(N-i,100));
		}
	}
	// we only need C[0] on the CPU
#pragma omp target update from(C[0:1])
	// we can now delete all the other gpu buffers
#pragma omp target exit data map(delete:A[0:N],B[0:N],C[0:N])
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] naive block func C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // block in a loop
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
		// Split outer loop among GPU compute units
#pragma omp target teams distribute
		for (int i=j; i<N; i+=100) {
			auto pC = &(C[i-j]);
			auto pA = &(A[i]);
			auto pB = &(B[i]);
			auto M = std::min(N-i,100);
			// vectorize the inner loop
#pragma omp parallel for simd
			for (int i=0; i<M; i++) oneI(pC[i],pA[i],pB[i]);
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

	{ // inline block function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
		// Split outer loop among GPU compute units
		// Note: Cannot use omp target teams distribute with a vector function, not valid syntax
#pragma omp target teams loop
		for (int i=j; i<N; i+=100) {
			// we are assuming the function is internally vectorized
			hundredI(&(C[i-j]), &(A[i]), &(B[i]), std::min(N-i,100));
		}
	}
	// we only need C[0] on the CPU
#pragma omp target update from(C[0:1])
	// we can now delete all the other gpu buffers
#pragma omp target exit data map(delete:A[0:N],B[0:N],C[0:N])
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] func block C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}


	{ // external block function
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
		// Split outer loop among GPU compute units
		// Note: Cannot use omp target teams distribute with a vector function, not valid syntax
#pragma omp target teams loop
		for (int i=j; i<N; i+=100) {
			// while the programmer may assume the function is vectorized
			// the compiler has no way to know it, so it assumes it is single threaded
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

	//
	// Just for compariso, use target teams loop instead of target distribute prarallel for

	{ // basic, no blocks, using loop omp syntax
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
		// We are less prescriptive here, compiler decides how to parallelize
#pragma omp target teams loop
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
	printf("[%i] loop plain C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
	}

	{ // block in a loop, using omp loop syntax
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A[0:N],B[0:N],C[0:N])
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will run on the GPU
		// Conceptually, we want to split outer loop among GPU compute units
		// We are less prescriptive here, compiler has more freedom
#pragma omp target teams loop
		for (int i=j; i<N; i+=100) {
			auto pC = &(C[i-j]);
			auto pA = &(A[i]);
			auto pB = &(B[i]);
			auto M = std::min(N-i,100);
			// We state compiler can vectorize the inner loop
#pragma omp loop bind(parallel)
			for (int i=0; i<M; i++) oneI(pC[i],pA[i],pB[i]);
		}
	}
	// we only need C[0] on the CPU
#pragma omp target update from(C[0:1])
	// we can now delete all the other gpu buffers
#pragma omp target exit data map(delete:A[0:N],B[0:N],C[0:N])
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] loop block C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
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
