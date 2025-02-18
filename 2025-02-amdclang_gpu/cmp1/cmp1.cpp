#include <stdio.h>
#include <string> 
#include <algorithm>
#include <stdlib.h>
#include <chrono>

#ifndef TFLOAT
#define TFLOAT float
#endif

int main(int argc, const char *argv[]) {
        if (argc!=3) {
		fprintf(stderr,"Error, wrong number of arguments\n");
		fprintf(stderr,"Usage:\n\t%s <size_multiplier> <comp_const>\n",argv[0]);
		return 1;
	}
	const uint32_t n_els = std::stol(argv[1])*240*1024+128; // 128-aligned, but not a multiple of 1k
	const TFLOAT start_val = std::stof(argv[2]);

	constexpr uint32_t n_comp = 3000*1024-128; // 128-aligned, but not a multiple of 1k
	TFLOAT *buf = new TFLOAT[n_els];

	auto t1 = std::chrono::high_resolution_clock::now();
#ifdef OMPGPU
#pragma omp target teams distribute parallel for map(from:buf[0:n_els])
#else
#pragma omp parallel for simd
#endif
	for (uint32_t i=0; i<n_els; i++) {
	  TFLOAT val = start_val+TFLOAT(0.00001)*(i/2) - TFLOAT(0.002)*(i%2);
	  TFLOAT m1 = TFLOAT(1.001);
	  TFLOAT m2 = TFLOAT(0.009);
	  for (uint32_t l=0; l<n_comp; l++) {
		  m1 += val*TFLOAT(0.00001);
		  val = val*m1 + TFLOAT(0.0003)*m2;
		  m2 -= val*TFLOAT(0.00002);
		  val = val*m2 - TFLOAT(0.0002)*m1;

	  }
	  buf[i] = val;
	}
	auto t2 = std::chrono::high_resolution_clock::now();

	std::sort(buf,buf+n_els);
	auto t3 = std::chrono::high_resolution_clock::now();
        printf("Result: %f\n",double(buf[0]));
        auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        auto time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("compute %.3f s sort %.3f s\n", time_span1.count(), time_span2.count());

	delete[] buf;
	return 0;
}
