#include <stdio.h>
#include <string> 
#include <algorithm>
#include <stdlib.h>
#include <chrono>

#ifndef TFLOAT
#define TFLOAT float
#endif

#ifndef VECBITS
#define VECBITS 512
#endif

#ifndef VECT
#define VECT (VECBITS/(8*sizeof(TFLOAT)))
#endif

typedef TFLOAT vfloat __attribute__ ((vector_size ( VECT*sizeof(TFLOAT) )));

typedef union {
        vfloat val;
        TFLOAT  arr[VECT];
} uTFLOAT;


int main(int argc, const char *argv[]) {
	constexpr uint32_t n_els = 240*1024+128; // 128-aligned, but not a multiple of 1k
	constexpr uint32_t n_comp = 3000*1024-128; // 128-aligned, but not a multiple of 1k
	vfloat *vbuf = new vfloat[n_els/VECT];
	TFLOAT *buf = (TFLOAT*)vbuf;
	TFLOAT start_val = std::stof(argv[1]);

	auto t1 = std::chrono::high_resolution_clock::now();
#ifdef OMPGPU
#pragma omp target teams distribute parallel for map(from:vbuf[0:(n_els/VECT)])
#else
#pragma omp parallel for simd
#endif
	for (uint32_t i=0; i<n_els; i+=VECT) {
          uTFLOAT u;
	  for (uint32_t a=0; a<VECT; a++) {
	     u.arr[a] = start_val+TFLOAT(0.00001)*((i+a)/2) - TFLOAT(0.002)*((i+a)%2);
	  }
	  vfloat val = u.val;
	  vfloat m1 = {TFLOAT(1.001)};
	  vfloat m2 = {TFLOAT(0.009)};
	  for (uint32_t l=0; l<n_comp; l++) {
		  m1 += val*TFLOAT(0.00001);
		  val = val*m1 + TFLOAT(0.0003)*m2;
		  m2 -= val*TFLOAT(0.00002);
		  val = val*m2 - TFLOAT(0.0002)*m1;

	  }
	  vbuf[i/VECT] = val;
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
