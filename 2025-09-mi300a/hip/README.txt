Using C++ and HIP to use AMD MI300A
===================================

Setup
=====
The following instructions work on SDSC Cosmos system.
Adapt as needed, if running on a different setup

All the compilers should be already in your path, so no additional setup needed.

Exercises
=========

1) Basic HIP example
--------------------
HIP is a GPU-centric extension of C++, that allows
for fine-grained vectorization of compute.

Here we look at a version of the STREAM Triad problem
that has both a OpenMP Target and HIP implementation,
so you can compare and contrast the different approach to
implement buffer addition.

There are two variations of the HIP algorithm, 
one simple and one slightly more optimized,
driven by pre-processor directives.

Look inside
triad_func.hip

The benchmarking main is in
triad_ext.cpp

Note that there is no CPU version for the HIP variants.

Try to build and execute it with
make triad
# Must explicity enable HSA_XNACK to get APU semantics
export HSA_XNACK=1
# Force the use of the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 ./triad_omp
taskset -c 24-47,120-143 ./triad_hip
taskset -c 24-47,120-143 ./triad_hip_opt

2) Using HIP for its advanced features
--------------------------------------
HIP exposes the programmer to a much more HW detail,
allowing for the use of features that are not exposed
to the OpenMP layer.

One example is GPU's shared memory, which can 
be quite beneficial when doing irregular memory
access patterns.

On the flip side, OpenMP providew convenience functionality,
like reductions, that are not avaialable in HIP.

Here we combine both, to show the tradeoffs of 
using the two approaches

The two variants are implemented in
permanova_func_omp.cpp
permanova_func_hip.hip

The benchmarking main is in
permanova.cpp

Note that there is no CPU version being benchmarked..

Try to build and execute it with
make permanova
# Must explicity enable HSA_XNACK to get APU semantics
export HSA_XNACK=1
# Force the use of the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 ./permanova

