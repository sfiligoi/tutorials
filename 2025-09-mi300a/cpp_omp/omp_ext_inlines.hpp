/*  Helper file, do not use directly
 *  Mimick a header-only library maintained by others
 *  that was made GPU aware.
 *
 *  Used by omp_ext.cpp
 */

#define oneM(c,a,b) \
	c += a*b; \
	a += 1.e-9*c; \
	b -= 1.e-10*c

// No need for omp declare target, assuming it will be inlined
static inline void oneI(float& c, float& a, float& b) {
	c += a*b;
	a += 1.e-9*c;
	b -= 1.e-10*c;
}

/*
 * We use a simple block loop as an example of a not-trivial logic
 *    one would want to abstract away into a function.
 * Real-life code would likely be significantly more complex.
 *
 */
// No need for omp declare target, assuming it will be inlined
static inline void hundredI(float* C, float* A, float* B, int N) {
#ifdef OMPGPU
#pragma omp parallel for 
#endif
	for (int i=0; i<N; i++) oneI(C[i],A[i],B[i]);
}

#ifdef OMPGPU
// Note: May generate a warning, but can be ignored
#endif

