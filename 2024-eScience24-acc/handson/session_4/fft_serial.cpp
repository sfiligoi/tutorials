/*  Simple serial FFT example
 *
 *  Compile with
 *  nvc++ -Mvect -o fft_serial fft_serial.cpp -lfftw3f
 */

#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <chrono>

void compute(const int N) {
	const int N2 = N/2+1;
	float* A = new float[N*N*100];
	float* B = new float[N*N*100];
	// complex use half the elements in fft
	fftwf_complex * cA = new fftwf_complex[N2*N*100];
	fftwf_complex * cB = new fftwf_complex[N2*N*100];

	auto t1 = std::chrono::high_resolution_clock::now();
  	auto plan_r2c = fftwf_plan_dft_r2c_2d(N,N,B,cB,FFTW_MEASURE);
	auto plan_c2r = fftwf_plan_dft_c2r_2d(N,N,cA,A,FFTW_MEASURE);

	// plan destroys the buffers, so initialize after the plan
	for (int i=0; i<(N*N*100); i++) {
		int r = rand();
		A[i] = 0.5+0.01*(r%100);
		B[i] = 1.5-0.001*(r%1000);
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<20; j++) {
		for (int i=0; i<100; i++) {
			// get my own 2D sub-buffer
		        float* pA = A+(N*N)*i;
		        float* pB = B+(N*N)*i;
        		fftwf_complex * pcA = cA+(N2*N)*i;
        		fftwf_complex * pcB = cB+(N2*N)*i;
	
			// compute FFT
			fftwf_execute_dft_r2c(plan_r2c,pA,pcA);
			fftwf_execute_dft_r2c(plan_r2c,pB,pcB);
			// mix the results
			for (int k=0; k<(N2*N); k++) {
				float a0 = pcA[k][0];
				float a1 = pcA[k][1];
				float b0 = pcB[k][0];
				float b1 = pcB[k][1];
				pcA[k][0] = (0.8*a0+0.2*b0)/N;
				pcA[k][1] = (0.7*a1+0.3*b1)/N;
				pcB[k][0] = (0.2*a0+0.9*b0)/N;
				pcB[k][1] = (0.3*a1+0.7*b1)/N;
			}
			// compute back FFT
			fftwf_execute_dft_c2r(plan_c2r,pcA,pA);
			fftwf_execute_dft_c2r(plan_c2r,pcB,pB);
			// mix the results
			for (int k=0; k<(N*N); k++) {
				float a = pA[k];
				float b = pB[k];
				pA[k] = (0.6*a+0.4*b)/N;
				pB[k] = (0.4*a+0.6*b)/N;
			}
		}
	}
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	float a = 0.0;
	for (int i=0; i<100; i++) a+= A[N*N*i+2*N+3];
	printf("[%i] A = %f (took %.3f s)\n", N,a, time_span.count());

	fftwf_destroy_plan(plan_c2r);
	fftwf_destroy_plan(plan_r2c);
	delete[] cB;
	delete[] cA;
	delete[] B;
	delete[] A;
}

int main(int argc, char* argv[]) {
	compute(50);
	compute(100);
	compute(200);
	compute(400);
	compute(800);
	// Too slow for the tutorial
	// compute(1600);
	printf("[%i] skipped\n",1600);

	return 0;
}
