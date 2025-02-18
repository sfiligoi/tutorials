Pure compute
============

This directory contains an example of pure CPP code that is completely compute-bound.

The starting file, `cmp1.cpp` contains the desired logic, and uses OpenMP for either CPU or GPU parallelization, based on compile parameters.

Unfortunately, not all compilers do a good job at optimizing the code. Both `clang++` and `nvc++` seem not to be able to properly vectorize the CPU code (tested `clang` 19 and `nvc++` 25), resulting in very slow execution.

The `cmp1_split.cpp` file tries to help the compiler with an intermediate loop, but it does not help at all (it actually breaks the `g++` optimization).

We thus proceed with [explicit vectorization](https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html) in the `cmp1_vect.cpp` file. This indeed works great for CPU code, but breaks GPU code generation for `nvc++` (tested with 25.1).

Note: Using a vector length longer than the HW support results in much faser execution with `clang++` (but not `g++`). By examining the ASM code, one can see that it is due to interleaving the compute loops, thus hiding the latency in the pipelined-CPU (single-threaded) execution.

We thus combine all that was learned above, and create `cmp1_uni.cpp`. This source code is always the fastest, although it does need different compilation arguments for different compilers (see `Makefile`).

Usage
-----

CPU-only compute with `g++`:
```
make all_gcc
./cmp1_gcc 1 3.2
./cmp1_split_gcc 1 3.2
./cmp1_vect_gcc 1 3.2
./cmp1_vectx4_gcc 1 3.2
./cmp1_uni_gcc 1 3.2
```

CPU-only compute with `clang++`:
```
make all_clang
./cmp1_clang 1 3.2
./cmp1_split_clang 1 3.2
./cmp1_vect_clang 1 3.2
./cmp1_vectx4_clang 1 3.2
./cmp1_uni_clang 1 3.2
```

GPU compute with `amdclang++`:
```
make all_amd_gpu
./cmp1_amd_gpu 1 3.2
./cmp1_vectx4_amd_gpu 1 3.2
./cmp1_uni_amd_gpu 1 3.2
```

GPU compute compute with `nvc++`:
```
#make all_nv_gpu
make cmp1_nv_gpu cmp1_uni_nv_gpu
./cmp1_nv_gpu 1 3.2
./cmp1_uni_nv_gpu 1 3.2
```

CPU-only compute with `nvc++`:
```
make all_nv_cpu
./cmp1_nv_cpu 1 3.2
./cmp1_uni_nv_cpu 1 3.2
```

