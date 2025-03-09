#include <stdio.h>
#include <string> 
#include <algorithm>
#include <random>
#include <stdlib.h>
#include <chrono>

#define TFLOAT float

int main(int argc, const char *argv[]) {
        if (argc!=3) {
		fprintf(stderr,"Error, wrong number of arguments\n");
		fprintf(stderr,"Usage:\n\t%s <size_multiplier> <seed>\n",argv[0]);
		return 1;
	}
	const uint32_t n_els = std::stol(argv[1])*240*1024+128; // 128-aligned, but not a multiple of 1k

	auto t0 = std::chrono::high_resolution_clock::now();
	// indexes to the next element, fixed 4G buffer
	uint32_t *idxs = new uint32_t[1000000000];
#pragma omp parallel for
	for (uint32_t i5=0; i5<5; i5++) { // do several parallel streams, as it is not trivial
	  std::mt19937 myRandomGenerator(std::stol(argv[2])+i5);
          for (uint32_t i=0; i<200000000; i++) idxs[i+i5*200000000] = myRandomGenerator() % 1000000000;
	}
#pragma omp target enter data map(to:idxs[0:1000000000])

	constexpr uint32_t n_comp = 4*1024-128; // 128-aligned, but not a multiple of 1k
	uint32_t *buf = new uint32_t[n_els];

	auto t1 = std::chrono::high_resolution_clock::now();
#ifdef OMPGPU
#pragma omp target teams distribute parallel for simd map(from:buf[0:n_els]) map(to:idxs[0:1000000000])
#else
#pragma omp parallel for simd
#endif
	for (uint32_t i=0; i<n_els; i++) {
	  uint32_t val = i;
	  for (uint32_t l=0; l<n_comp; l++) {
		  val = idxs[val]; // find the next location using the current one

	  }
	  buf[i] = val;
	}
	auto t2 = std::chrono::high_resolution_clock::now();

	// sort is here just to make all the buffer relevant
	std::sort(buf,buf+n_els);
	auto t3 = std::chrono::high_resolution_clock::now();
        printf("Result: %i\n",int(buf[0]));
        auto time_span0 = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
        auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        auto time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("compute %.3f s init %.3f sort %.3f s\n", time_span1.count(), time_span0.count(), time_span2.count());

	delete[] buf;
#pragma omp target exit data map(release:idxs[0:1000000000])
	delete[] idxs;
	return 0;
}
