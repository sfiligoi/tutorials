/*  Helper file, do not use directly
 *  Mimick a library maintained by others
 *
 *  Used by full_ext.cpp
 */

static inline void oneI(float& c, float& a, float& b) {
	c += a*b;
	a += 1.e-9*c;
	b -= 1.e-10*c;
}

/*
 * To be invoked from CPU serial code
 * but will parallelize internally.
 *
 * The function is very simple, for demonstration purposes only.
 * Real-life code would likely be significantly more complex.
 */
void compute_range(int start, int end, float* A, float* B, float* C) {
#ifdef OMPGPU
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
	for (int i=start; i<end; i++) {
		oneI(C[i-start], A[i], B[i]);
	}
}

