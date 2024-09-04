/*  Batch multi-threaded FFT example
 *
 *  Compile with
 *  nvc++ -Mvect -mp -o fft_batch fft_batch.cpp -lfftw3f_threads -lfftw3f
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <chrono>

void compute(const int N) {
	const int N2 = N/2+1;
	float* AB = new float[2*N*N*100];
	// keep them consecutive, so we can compute together
	float* A = AB;
	float* B = AB+(N*N*100);
	// complex use half the elements in fft
	fftwf_complex * cAB = new fftwf_complex[2*N2*N*100];
	fftwf_complex * cA = cAB;
	fftwf_complex * cB = cAB+(N2*N*100);

	auto t1 = std::chrono::high_resolution_clock::now();
	int r2c_dim[2] = {N,N};
	auto plan_r2c_many = fftwf_plan_many_dft_r2c(
          2, r2c_dim, 200, 
          AB, NULL, 1, N*N,
	  cAB, NULL, 1, N2*N,
          FFTW_MEASURE);

	int c2r_dim[2] = {N,N};
	auto plan_c2r_many = fftwf_plan_many_dft_c2r(
          2, c2r_dim, 200, 
          cAB, NULL, 1, N2*N,
	  AB, NULL, 1, N*N,
          FFTW_MEASURE);

	// plan destroys the buffers, so initialize after the plan
	for (int i=0; i<(N*N*100); i++) {
		int r = rand();
		A[i] = 0.5+0.01*(r%100);
		B[i] = 1.5-0.001*(r%1000);
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<20; j++) {
		// compute all ffts at once
		fftwf_execute_dft_r2c(plan_r2c_many,AB,cAB);
		// mix the results
#pragma omp parallel for
		for (int k=0; k<100*(N2*N); k++) {
			float a0 = cA[k][0];
			float a1 = cA[k][1];
			float b0 = cB[k][0];
			float b1 = cB[k][1];
			cA[k][0] = (0.8*a0+0.2*b0)/N;
			cA[k][1] = (0.7*a1+0.3*b1)/N;
			cB[k][0] = (0.2*a0+0.9*b0)/N;
			cB[k][1] = (0.3*a1+0.7*b1)/N;
		}
		// compute all ffts at once
		fftwf_execute_dft_c2r(plan_c2r_many,cAB,AB);
		// mix the results
#pragma omp parallel for
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
	for (int i=0; i<100; i++) a+= A[N*N*i+2*N+3];
	printf("[%i] A = %f (took %.3f s)\n", N,a, time_span.count());

	fftwf_destroy_plan(plan_c2r_many);
	fftwf_destroy_plan(plan_r2c_many);
	delete[] cAB;
	delete[] AB;
}

int main(int argc, char* argv[]) {
	// artificially limit, to be nice to other tutorial users
	omp_set_num_threads(10);
	fftwf_plan_with_nthreads(10);

	compute(50);
	compute(100);
	compute(200);
	compute(400);
	compute(800);
	compute(1600);

	return 0;
}
