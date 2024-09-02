/*  Helper file, do not use directly
 *
 *  See loop_gpu_func.cpp
 */

void oneE(float& c, float& a, float& b) {
	c += a*b;
	a += 1.e-9*c;
	b -= 1.e-10*c;
}

#pragma omp declare target to(oneE)

/*
 * We use a simple block loop as an example of a not-trivial logic
 *    one would want to abstract away into a function.
 * Real-life code would likely be significantly more complex.
 *
 */
void hundredE(float* C, float* A, float* B, int N) {
	for (int i=0; i<N; i++) oneE(C[i],A[i],B[i]);
}

#pragma omp declare target to(hundredE)

