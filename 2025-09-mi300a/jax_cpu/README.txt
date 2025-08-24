Using JAX to speed up Python functions on the CPU
=================================================

Setup
=====
The following instructions work on SDSC Cosmos system.
Adapt as needed, if running on a different setup

Assuming you are inside a conda environment,
install cupy and some support libraries:
----------------------------------------
conda create -n jax-cpu-tutorial -c conda-forge python=3.12 jax scikit-bio wget
conda activate jax-cpu-tutorial

Fetch large DistanceMatrix file, used in examples:
--------------------------------------------------
wget http://uaf-10.t2.ucsd.edu/~sfiligoi/unifrac_inputs/emp/uw_emp.h5

Exercises
=========

1) Use JAX's Just-In-Time compiler
----------------------------------
A numpy array can be converted to a JAX array,
and then you operate on it as before.
By default, JAX compiles each step independently.
Compiling a whole function can result in additional speedups.

Look inside
center_matrix_jit.py

and then try to execute it with
# Force the use of the 2nd APU on the node
taskset -c 24-47,120-143 python center_matrix_jit.py

