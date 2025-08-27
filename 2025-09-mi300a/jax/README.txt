Using JAX to speed up compute on AMD MI300A
============================================

Setup
=====
The following instructions work on SDSC Cosmos system.
Adapt as needed, if running on a different setup

Assuming you are inside a conda environment,
install cupy and some support libraries:
----------------------------------------
conda create -n jax-tutorial -c conda-forge python=3.12 gxx scikit-bio wget make
conda activate jax-tutorial
# AMD-GPU enabled cupy not avaialble in conda, use pip
export LLVM_PATH=/opt/rocm/llvm
pip install 'jax[rocm]'

Fetch large DistanceMatrix file, used in examples:
--------------------------------------------------
wget http://uaf-10.t2.ucsd.edu/~sfiligoi/unifrac_inputs/emp/uw_emp.h5

Exercises
=========

1) Compare numpy vs JAX performance
------------------------------------
A numpy array can be converted to a JAX array,
and then you operate on it as before.

Look inside
center_matrix_naive.py

and then try to execute it with
# Force the use of the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 python center_matrix_naive.py

2) Use JAX's Just-In-Time compiler
----------------------------------
By default, JAX compiles each step independently.
Compiling a whole function can result in additional speedups.

Look inside
center_matrix_jit.py

and then try to execute it with
# Force the use of the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 python center_matrix_jit.py

3) JAX linalg functions
-----------------------
JAX provides several NumPy equivalent functions.

This file provides an example
linalg_jax.py

Look inside it, then Try to execute it with
# Force the use of the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 python linalg_jax.py 

4) Mixing NumPy and JAX
-----------------------
Not all library functions are JAX-aware.
For those, you may need to do a conversion
between the two format.

Look inside
linalg_mixed.py

and then try to execute it with
# Force the use of the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 python linalg_mixed.py

