/*  Helper file, do not use directly
 *  Mimick a header-only library maintained by others
 *
 *  Used by cpp_ext.cpp
 */

#define oneM(c,a,b) \
	c += a*b; \
	a += 1.e-9*c; \
	b -= 1.e-10*c

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
static inline void hundredI(float* C, float* A, float* B, int N) {
	for (int i=0; i<N; i++) oneI(C[i],A[i],B[i]);
}

