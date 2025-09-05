//  Simple OpenMP loop example with function calling

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>

// Just declare here to avoid using a header file
// implementation in full_ext_func.cpp
// In real libraries there would have been a header file for this
void compute_range(int start, int end, float* A, float* B, float* C);


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

	{
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
	for (int i=0; i<N; i++) { A[i]=oA[i]; B[i]=oB[i]; C[i]=oC[i];}
	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// we really don't know what happens inside compute_range
		// but we hope it can be parallelized
		compute_range(j, N, A, B, C);
	}
	auto t3 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	// Printing C[0] to avoid over-optimization by the compiler
	printf("[%10i try %i] full ext C0 = %8.3f (took %.3f s)\n", N,li, C[0], time_span.count());
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
