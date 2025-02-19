#include <stdio.h>
#include <string> 
#include <algorithm>
#include <stdlib.h>
#include <chrono>

#ifndef TFLOAT
#define TFLOAT float
#endif

#ifndef VECT
#define VECT 1
#endif

#if ( VECT > 1 )
// use vector registers
typedef TFLOAT vfloat __attribute__ ((vector_size ( VECT*sizeof(TFLOAT) )));
#else
// if VEC==1, just use the regular scalar logic
typedef TFLOAT vfloat;
#endif

typedef union {
        vfloat val;
        TFLOAT arr[VECT];
} uTFLOAT;


int main(int argc, const char *argv[]) {
        if (argc!=3) {
		fprintf(stderr,"Error, wrong number of arguments\n");
		fprintf(stderr,"Usage:\n\t%s <size_multiplier> <comp_const>\n",argv[0]);
		return 1;
	}
	const uint32_t n_els = std::stol(argv[1])*240*1024+128; // 128-aligned, but not a multiple of 1k
	const TFLOAT start_val = std::stof(argv[2]);

	constexpr uint32_t n_comp = 3000*1024-128; // 128-aligned, but not a multiple of 1k
	vfloat *vbuf = new vfloat[n_els/VECT];
	TFLOAT *buf = (TFLOAT*)vbuf;

	auto t1 = std::chrono::high_resolution_clock::now();
#ifdef OMPGPU
#pragma omp target teams distribute parallel for map(from:vbuf[0:(n_els/VECT)])
#else
#pragma omp parallel for simd
#endif
	for (uint32_t i=0; i<n_els; i+=(VECT*2)) {
	  // using 2 of everything to hide latency in pipelined compute
          uTFLOAT u_a;
          uTFLOAT u_b;
	  for (uint32_t a=0; a<VECT; a++) {
	     u_a.arr[a] = start_val+TFLOAT(0.00001)*((i+a)/2)        - TFLOAT(0.002)*((i+a)%2);
	     u_b.arr[a] = start_val+TFLOAT(0.00001)*((i+a+VECT)/2)   - TFLOAT(0.002)*((i+a+VECT)%2);
	  }
	  vfloat val_a = u_a.val;
	  vfloat val_b = u_b.val;
	  vfloat m1_a = {TFLOAT(1.001)};
	  vfloat m1_b = {TFLOAT(1.001)};
	  vfloat m2_a = {TFLOAT(0.009)};
	  vfloat m2_b = {TFLOAT(0.009)};
	  for (uint32_t l=0; l<n_comp; l++) {
		  m1_a += val_a*TFLOAT(0.00001);
		  m1_b += val_b*TFLOAT(0.00001);
		  val_a = val_a*m1_a + TFLOAT(0.0003)*m2_a;
		  val_b = val_b*m1_b + TFLOAT(0.0003)*m2_b;
		  m2_a -= val_a*TFLOAT(0.00002);
		  m2_b -= val_b*TFLOAT(0.00002);
		  val_a = val_a*m2_a - TFLOAT(0.0002)*m1_a;
		  val_b = val_b*m2_b - TFLOAT(0.0002)*m1_b;

	  }
	  vbuf[i/VECT]   = val_a;
	  vbuf[i/VECT+1] = val_b;
	}
	auto t2 = std::chrono::high_resolution_clock::now();

	// sort is here just to make all the buffer relevant
	std::sort(buf,buf+n_els);
	auto t3 = std::chrono::high_resolution_clock::now();
        printf("Result: %f\n",double(buf[0]));
        auto time_span1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        auto time_span2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
	printf("compute %.3f s sort %.3f s\n", time_span1.count(), time_span2.count());

	delete[] buf;
	return 0;
}
