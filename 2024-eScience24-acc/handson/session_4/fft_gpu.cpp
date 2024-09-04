/*  Simple multi-threaded (mt+OpenMP) FFT example
 *
 *  Compile with
 *  nvc++ -mp=gpu -o fft_gpu fft_gpu.cpp -cudalib=cufft
 */

#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <chrono>

void compute(const int N) {
	const int N2 = N/2+1;
	const int Ns200 = 2*N*N*100;
	const int N2s200 = 2*N2*N*100;

	float* AB = new float[2*N*N*100];
	// keep them consecutive, so we can compute together
	float* A = AB;
	float* B = AB+(N*N*100);
	// complex use half the elements in fft
	cuComplex * cAB = new cuComplex[2*N2*N*100];
	cuComplex * cA = cAB;
	cuComplex * cB = cAB+(N2*N*100);

	auto t1 = std::chrono::high_resolution_clock::now();
	int r2c_dim[2] = {N,N};
	cufftHandle plan_r2c_many;
	cufftPlanMany(&plan_r2c_many,
          2, r2c_dim, 
          NULL, 1, N*N,
	  NULL, 1, N2*N,
          CUFFT_R2C, 200);

	int c2r_dim[2] = {N,N};
	cufftHandle plan_c2r_many;
	cufftPlanMany(&plan_c2r_many,
          2, c2r_dim, 
          NULL, 1, N2*N,
	  NULL, 1, N*N,
          CUFFT_C2R, 200);

	for (int i=0; i<(N*N*100); i++) {
		int r = rand();
		A[i] = 0.5+0.01*(r%100);
		B[i] = 1.5-0.001*(r%1000);
	}

	auto t2 = std::chrono::high_resolution_clock::now();

	// all compute happens on the GPU, so move the data there
#pragma omp target enter data map(to:AB[0:Ns200],cAB[0:N2s200])

	// this must be serial, due to loop dependency
	for (int j=0; j<20; j++) {
		// tell the compilers to use the GPU pointers when invoking functions
#pragma omp target data use_device_ptr(AB,cAB)
		{
			// compute all ffts at once on the GPU
			cufftExecR2C(plan_r2c_many,AB,cAB);
		}
		// mix the results
#pragma omp target teams distribute parallel for simd 
		for (int k=0; k<100*(N2*N); k++) {
			float a0 = cA[k].x;
			float a1 = cA[k].y;
			float b0 = cB[k].x;
			float b1 = cB[k].y;
			cA[k].x = (0.8*a0+0.2*b0)/N;
			cA[k].y = (0.7*a1+0.3*b1)/N;
			cB[k].x = (0.2*a0+0.9*b0)/N;
			cB[k].y = (0.3*a1+0.7*b1)/N;
		}
#pragma omp target data use_device_ptr(AB,cAB)
		{
			// compute all ffts at once on the GPU
			cufftExecC2R(plan_c2r_many,cAB,AB);
		}
		// mix the results
#pragma omp target teams distribute parallel for simd 
		for (int k=0; k<100*(N*N); k++) {
			float a = A[k];
			float b = B[k];
			A[k] = (0.6*a+0.4*b)/N;
			B[k] = (0.4*a+0.6*b)/N;
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	float a = 0.0;
#pragma omp target teams distribute parallel for simd reduction(+:a)
	for (int i=0; i<100; i++) a+= A[N*N*i+2*N+3];

	printf("[%i] A = %f (took %.3f s)\n", N,a, time_span.count());

#pragma omp target exit data map(delete:AB[0:Ns200],cAB[0:N2s200])

	cufftDestroy(plan_c2r_many);
	cufftDestroy(plan_r2c_many);
	delete[] cAB;
	delete[] AB;
}

int main(int argc, char* argv[]) {
	compute(50);
	compute(100);
	compute(200);
	compute(400);
	compute(800);
	compute(1600);

	return 0;
}
