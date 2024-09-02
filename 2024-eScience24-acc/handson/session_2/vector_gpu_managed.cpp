/*  Simple fixed-size vector in C++ loop on GPU example (OpenMP)
 *
 *  Compile with
 *  nvc++ -mp=gpu -gpu=managed -o vector_gpu_managed vector_gpu_managed.cpp
 *  Note, fill fail with
 *  nvc++ -mp=gpu -o vector_gpu_full vector_gpu_managed.cpp
 */

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
        // all compute happens on the GPU, we try to move the data there
	// but this only moves the object metadata, not the actual buffer
#pragma omp target enter data map(to:A,B,C)


	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will be parallelized on the GPUa
		// Requires NVIDIA managed memory to handle the internal buffers
#pragma omp target teams distribute parallel for simd
		for (int i=j; i<N; i++) {
			C[i-j] += A[i]*B[i];
			A[i] += 1.e-9*C[i-j];
			B[i] -= 1.e-10*C[i-j];
		}
	}
	// nothing changed to the metadata, so just delete
#pragma omp target exit data map(delete:A,B,C)
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
        // all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:A,B,C)

	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will be parallelized on the GPU
		// Since arrays are not dynamic, we do not need NVIDIA managed memory
#pragma omp target teams distribute parallel for simd
		for (int i=j; i<N; i++) {
			C[i-j] += A[i]*B[i];
			A[i] += 1.e-9*C[i-j];
			B[i] -= 1.e-10*C[i-j];
		}
	}
	// Bring back the results to CPU
	// Since we cannot access internals, we transfer all of it
#pragma omp target exit data map(from:C)
	// We can just destroy the rest
#pragma omp target exit data map(delete:A,B)
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] C0 = %f (took %.3f s)\n", N,C[0], time_span.count());
}

int main(int argc, char* argv[]) {
	compute2<10000000>();
	// Warning: will fail without managed memory
	compute(1000);
	compute(10000);
	compute(100000);
	compute(1000000);
	compute(10000000);
	compute(100000000);

	return 0;
}
