/*  Helper file, do not use directly
 *
 *  See loop_cpu_func.cpp
 */

void oneE(float& c, float& a, float& b) {
	c += a*b;
	a += 1.e-9*c;
	b -= 1.e-10*c;
}

void hundredE(float* C, float* A, float* B, int N) {
	for (int i=0; i<N; i++) oneE(C[i],A[i],B[i]);
}

