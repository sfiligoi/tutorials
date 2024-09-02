/*  Helper file, do not use directly
 *
 *  See block_gpu_func.cpp
 */

void oneE(float& c, float& a, float& b) {
	c += a*b;
	a += 1.e-9*c;
	b -= 1.e-10*c;
}

/*
 * We use a simple block loop as an example of a not-trivial logic
 *    one would want to abstract away into a function.
 * Real-life code would likely be significantly more complex.
 *
 * Note: While this will compile without any errors, or even warnings,
 *       the caller has no way to know that the function is supposed to be vectorized.
 */
void hundredE(float* C, float* A, float* B, int N) {
#pragma omp loop bind(parallel)
	for (int i=0; i<N; i++) oneE(C[i],A[i],B[i]);
}

#pragma omp declare target to(hundredE)

