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

The mbenchmarking main is in
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

