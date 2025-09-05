/*  Helper file, do not use directly
 *  Adds the GPU-specific pieces to make
 *  cpp_ext_func.cpp
 *  usable from GPU loops.
 */

#include "cpp_ext_func.cpp"

// the following will force the creation of a GPU variant
#pragma omp declare target to(oneE)
#pragma omp declare target to(hundredE)

/*
 * Note: The compiler may throw a warning, but it is harmless.
 */

