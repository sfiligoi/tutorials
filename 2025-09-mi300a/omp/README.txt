Using C++ OpenMP to use AMD MI300A
==================================

Setup
=====
The following instructions work on SDSC Cosmos system.
Adapt as needed, if running on a different setup

All the compilers should be already in your path, so no additional setup needed.

Exercises
=========

1) Basic OpenMP and evaluate cost of dynamic memory
---------------------------------------------------
This example provides a basic example on how to
initialize a large buffer either serially on CPU,
in parallel on the CPU using OpenMP, and
fully on GPU using OpenMP Target offload.

It also showcases the cost of using dynamic memory 
with GPU code, and how that differs from CPU-only codes.

Look inside
dynamic_memory.cpp

and then try to build and execute it with
make dynamic_memory
# Force the use of the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
# Must explicity enable HSA_XNACK to get APU semantics
HSA_XNACK=1 taskset -c 24-47,120-143 ./dynamic_memory

2) Conditional compilation for CPU vs GPU
-----------------------------------------
It is often desirable to keep the same codebase for both CPU and GPU platforms.
We can achieve this by using the compiler preprocessor.

We also showcase how much faster is GPU compute
(excluding initialization)
when performing the classic traid compute.

Look inside
triad.cpp

and then try to build and execute it with
make triad
# Force the use of the 2nd APU on the node
taskset -c 24-47,120-143 ./triad_cpu
export ROCR_VISIBLE_DEVICES=1
# Must explicity enable HSA_XNACK to get APU semantics
HSA_XNACK=1 taskset -c 24-47,120-143 ./triad_gpu

3) Different optimization options
---------------------------------
Memory-heavy algorithms benefit most from
avoiding intermediate temporary buffers.

But there may be further savings to be had if using the caches,
although that may come at a vectorization cost.
Given the architectural differences between CPU and GPU compute,
a different optimization may be best in the two cases.

Look inside
center_matrix_omp.cpp

and then try to build and execute it with
make center_matrix
# Force the use of the 2nd APU on the node
taskset -c 24-47,120-143 ./center_matrix_cpu
export ROCR_VISIBLE_DEVICES=1
# Must explicity enable HSA_XNACK to get APU semantics
HSA_XNACK=1 taskset -c 24-47,120-143 ./center_matrix_gpu


4) Mixing serial and parallel code
---------------------------------
Sometimes, an algorithm just does not lend itself
to being implementated in a parallel way.

If the most costly part of the problem solving can still
be executed in parallel, parallelizing just those parts
can still provide a significnat speedup.

And whatever is parallelized, can typically benefit from GPU acceleration.
While all serial code stays on the CPU.

Here is a synthetic benchmark that showcases this.
Look inside
partial_parallel.cpp

and then try to build and execute it with
make partial_parallel
# Force the use of the 2nd APU on the node
taskset -c 24-47,120-143 ./partial_parallel_cpu
export ROCR_VISIBLE_DEVICES=1
# Must explicity enable HSA_XNACK to get APU semantics
HSA_XNACK=1 taskset -c 24-47,120-143 ./partial_parallel_gpu

