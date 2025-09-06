//  Simple FFT example emphasizing batching

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <stdexcept>

#include <hipfft/hipfft.h>
#include <hip/hip_runtime_api.h>

// compute a single FFT
void compute_one(const int N) {
   const int N2 = N/2+1;

   float* mat = new float[N*N];
   // complex use half the elements in fft
   hipComplex * cmat = new hipComplex[N2*N];

   hipfftHandle plan_r2c, plan_c2r;
   // not benchmarking the setup
   {
	int r2c_dim[2] = {N,N};
	hipfftPlanMany(&plan_r2c,
          2, r2c_dim, 
          NULL, 1, N*N,
	  NULL, 1, N2*N,
          HIPFFT_R2C, 1);

	int c2r_dim[2] = {N,N};
	hipfftPlanMany(&plan_c2r,
          2, c2r_dim, 
          NULL, 1, N2*N,
	  NULL, 1, N*N,
          HIPFFT_C2R, 1);
   }
   for (int li=1; li<=2; li++) {
#pragma omp target teams distribute parallel for collapse(2)
	for (uint32_t row=0; row<N; row++) {
	  for (uint32_t col=0; col<N; col++) {
            mat[N*row+col] = 0.1*sin(row*1.1)+0.15*cos(col*0.9);
	  }
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	// forward FFT
	hipfftExecR2C(plan_r2c,mat,cmat);

        if (hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed");
	// filter
#pragma omp target teams distribute parallel for collapse(2)
	for (uint32_t row=(N/3); row<(2*N/3); row++) {
	  for (uint32_t col=(N2/3); col<(2*N2/3); col++) {
            cmat[N2*row+col] = make_hipFloatComplex(0.0f, 0.0f);
	  }
	}
	
	// reverse FFT
	hipfftExecC2R(plan_c2r,cmat,mat);
        if (hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed");
	auto t2 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

	// print out one element, just to avoid 
	printf("[%6ix%6ix 1 try %i] took %.5f s [Val: %.2f]\n", N,N, li,time_span.count(), mat[2]);
   }

   hipfftDestroy(plan_c2r);
   hipfftDestroy(plan_r2c);
   delete[] cmat;
   delete[] mat;
}


// compute many FFTs, still use non-batched approach
void compute_many_seq(const int N, const int rep) {
   const int N2 = N/2+1;

   float* mat_many = new float[uint64_t(rep)*N*N];
   // complex use half the elements in fft
   hipComplex * cmat = new hipComplex[N2*N];

   hipfftHandle plan_r2c, plan_c2r;
   // not benchmarking the setup
   {
	int r2c_dim[2] = {N,N};
	hipfftPlanMany(&plan_r2c,
          2, r2c_dim, 
          NULL, 1, N*N,
	  NULL, 1, N2*N,
          HIPFFT_R2C, 1);

	int c2r_dim[2] = {N,N};
	hipfftPlanMany(&plan_c2r,
          2, c2r_dim, 
          NULL, 1, N2*N,
	  NULL, 1, N*N,
          HIPFFT_C2R, 1);
   }
   for (int li=1; li<=2; li++) {
	// we have all the input matrices in buffer at the beginning
#pragma omp target teams distribute parallel for collapse(3)
	for (uint32_t r=0; r<rep; r++) {
	 for (uint32_t row=0; row<N; row++) {
	  for (uint32_t col=0; col<N; col++) {
            mat_many[uint64_t(N*N)*r + N*row + col] = 0.1*sin(r*0.01+row*1.1)+0.15*cos(col*0.9-r*0.02);
	  }
	 }
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	for (uint32_t r=0; r<rep; r++) { // we process them one at a time
   	  float* mat = mat_many+uint64_t(N*N)*r;
	  // I can use the same cmat in all iterations

	  // forward FFT
	  hipfftExecR2C(plan_r2c,mat,cmat);

          if (hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed");
	  // filter
#pragma omp target teams distribute parallel for collapse(2)
	  for (uint32_t row=(N/3); row<(2*N/3); row++) {
	    for (uint32_t col=(N2/3); col<(2*N2/3); col++) {
              cmat[N2*row+col] = make_hipFloatComplex(0.0f, 0.0f);
	    }
	  }
	
	  // reverse FFT
	  hipfftExecC2R(plan_c2r,cmat,mat);
          if (hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed");
	}
	auto t2 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

	// print out one element, just to avoid 
	printf("[%6ix%6ix%4i try %i] took %.3f s [Val: %.2f]\n", N,N, rep, li,time_span.count(), mat_many[2]);
   }

   hipfftDestroy(plan_c2r);
   hipfftDestroy(plan_r2c);
   delete[] cmat;
   delete[] mat_many;
}

// compute many FFTs, use thebatched interface
void compute_many(const int N, const int rep) {
   const int N2 = N/2+1;

   float* mat_many = new float[uint64_t(rep)*N*N];
   // complex use half the elements in fft
   hipComplex * cmat_many = new hipComplex[uint64_t(rep)*N2*N];

   hipfftHandle plan_r2c_many, plan_c2r_many;
   // not benchmarking the setup
   {
	int r2c_dim[2] = {N,N};
	hipfftPlanMany(&plan_r2c_many,
          2, r2c_dim, 
          NULL, 1, N*N,
	  NULL, 1, N2*N,
          HIPFFT_R2C, rep);

	int c2r_dim[2] = {N,N};
	hipfftPlanMany(&plan_c2r_many,
          2, c2r_dim, 
          NULL, 1, N2*N,
	  NULL, 1, N*N,
          HIPFFT_C2R, rep);
   }
   for (int li=1; li<=2; li++) {
	 // we have all input matrices in memory
#pragma omp target teams distribute parallel for collapse(3)
	for (uint32_t r=0; r<rep; r++) {
	 for (uint32_t row=0; row<N; row++) {
	  for (uint32_t col=0; col<N; col++) {
            mat_many[uint64_t(N*N)*r + N*row + col] = 0.1*sin(r*0.01+row*1.1)+0.15*cos(col*0.9-r*0.02);
	  }
	 }
	}

	auto t1 = std::chrono::high_resolution_clock::now();
	// we process them all in one pass

	// Unlike the seq version, we need cmat to conatin all the matrices

	// forward FFT
	hipfftExecR2C(plan_r2c_many,mat_many,cmat_many);

        if (hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed");
	// filter
#pragma omp target teams distribute parallel for collapse(3)
	for (uint32_t r=0; r<rep; r++) {
	 for (uint32_t row=(N/3); row<(2*N/3); row++) {
	  for (uint32_t col=(N2/3); col<(2*N2/3); col++) {
            cmat_many[uint64_t(N2*N)*r + N2*row+col] = make_hipFloatComplex(0.0f, 0.0f);
	  }
	 }
	}
	
	// reverse FFT
	hipfftExecC2R(plan_c2r_many,cmat_many,mat_many);
        if (hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("hipDeviceSynchronize failed");
	auto t2 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

	// print out one element, just to avoid 
	printf("[%6ix%6ix%4i try %i] took %.3f s [Val: %.2f]\n", N,N, rep, li,time_span.count(), mat_many[2]);
   }

   hipfftDestroy(plan_c2r_many);
   hipfftDestroy(plan_r2c_many);
   delete[] cmat_many;
   delete[] mat_many;
}

int main(int argc, char* argv[]) {
	printf("--- Single FFT\n");
	compute_one(8*9*7);
	compute_one(64*9*7);
	compute_one(9*64*9*7);
	printf("--- Multiple FFTs in sequence\n");
	compute_many_seq(8*9*7,1000);
	compute_many_seq(64*9*7,100);
	compute_many_seq(9*64*9*7,10);
	printf("--- Multiple FFTs in batch mode\n");
	compute_many(8*9*7,1000);
	compute_many(64*9*7,100);
	compute_many(9*64*9*7,10);

	return 0;
}
