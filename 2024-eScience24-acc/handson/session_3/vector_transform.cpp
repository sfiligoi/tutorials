/*  Simple fixed-size vector transform example - works both in multi-core and GPU mode, depending on compiler options
 *
 *  Compile with
 *  nvc++ --c++17 -Mvect -stdpar=multicore -o vector_transform vector_transform.cpp
 *  or
 *  nvc++ --c++17 -stdpar=gpu -o vector_transform vector_transform.cpp
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <chrono>
#include <algorithm>
#include <execution>

typedef struct {
	float A;
	float B;
} Tab;

void compute(const int N) {
	std::vector<Tab> AB(N);
	std::vector<float> C(N);

	auto t1 = std::chrono::high_resolution_clock::now();
	for (int i=0; i<N; i++) {
		int r = rand();
		AB[i].A = 0.5+0.01*(r%100);
		AB[i].B = 1.5-0.001*(r%1000);
		C[i] = 0.0;
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	// this must be serial, due to loop dependency
	for (int j=0; j<100; j++) {
		// this loop will be parallelized
		// std::execution::par_unseq tells compiler it can aggressively parallelize/vectorize
		std::transform(std::execution::par_unseq,
			AB.data()+j, AB.data()+N, C.begin(), C.begin(), [=] (const Tab ab, const float c) -> float {
			return c+ab.A*ab.B;
		});
		std::transform(std::execution::par_unseq,
			AB.data()+j, AB.data()+N, C.begin(), AB.data()+j, [=] (const Tab ab, const float c) -> Tab {
			Tab res;
			res.A = ab.A + 1.e-9*c;
			res.B = ab.B + 1.e-10*c;
			return res;
		});
	}
	auto t3 = std::chrono::high_resolution_clock::now();

        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("[%i] C0 = %f (took %.4f s)\n", N,C[0], time_span.count());
}


int main(int argc, char* argv[]) {
	// artificially limit, to be nice to other tutorial users
	omp_set_num_threads(10);

	compute(1000);
	compute(10000);
	compute(100000);
	compute(1000000);
	compute(10000000);
	compute(100000000);

	return 0;
}
