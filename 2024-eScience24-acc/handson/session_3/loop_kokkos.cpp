/*  Simple Kokkos-based loop example
 *
 *  Compile with
 *  cmake -B build_omp -DKokkos_ENABLE_OPENMP=ON && cd build_omp && make
 *  or
 *  cmake -B build_cuda -DKokkos_ENABLE_CUDA=ON && cd build_cuda && make
 */

#include <stdio.h>
#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include <chrono>

void compute(const int N) {
	// use Kokkos views to properly handle partitioned memory
	Kokkos::View<float*> A("A",N);
	Kokkos::View<float*> B("B",N);
	Kokkos::View<float*> C("C",N);

	// Mirrors are really what makes it possible
	typename Kokkos::View<float *>::HostMirror Acpu  = Kokkos::create_mirror_view(A); 
	typename Kokkos::View<float *>::HostMirror Bcpu  = Kokkos::create_mirror_view(B); 
	typename Kokkos::View<float *>::HostMirror Ccpu  = Kokkos::create_mirror_view(C); 

	auto t1 = std::chrono::high_resolution_clock::now();
	// Initialize the mirror
	for (int i=0; i<N; i++) {
		int r = rand();
		Acpu(i) = 0.5+0.01*(r%100);
		Bcpu(i) = 1.5-0.001*(r%1000);
		Ccpu(i) = 0.0;
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	// copy buffers to GPU
	Kokkos::deep_copy (A, Acpu);
	Kokkos::deep_copy (B, Bcpu);
	Kokkos::deep_copy (C, Ccpu);
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// Make this loop parallel
		Kokkos::parallel_for(N-j, 
		    KOKKOS_LAMBDA (const int i) {
			C(i) += A(i+j)*B(i+j);
			A(i+j) += 1.e-9*C(i);
			B(i+j) -= 1.e-10*C(i);
		});
	}
	auto t3 = std::chrono::high_resolution_clock::now();

	// copy result back to the CPU
	Kokkos::deep_copy (Ccpu, C);
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] C0 = %f (took %.4f s)\n", N,Ccpu[0], time_span.count());
}

int main(int argc, char* argv[]) {
  	Kokkos::initialize(argc, argv);

	compute(1000);
	compute(10000);
	compute(100000);
	compute(1000000);
	compute(10000000);
	compute(100000000);

  	Kokkos::finalize();
	return 0;
}
