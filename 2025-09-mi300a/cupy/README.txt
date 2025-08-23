Using cuPy to speed up compute on AMD MI300A
============================================

Setup
=====
The following instructions work on SDSC Cosmos system.
Adapt as needed, if running on a different setup

Assuming you are inside a conda environment,
install cupy and some support libraries:
----------------------------------------
conda create -n cupy-tutorial -c conda-forge python=3.12 gxx scikit-bio numpy-allocator wget make
conda activate cupy-tutorial
# AMD-GPU enabled cupy not avaialble in conda, build using pip
export ROCM_HOME=/opt/rocm
export CUPY_INSTALL_USE_HIP=1
pip install cupy

Fetch large DistanceMatrix file, used in examples:
--------------------------------------------------
wget http://uaf-10.t2.ucsd.edu/~sfiligoi/unifrac_inputs/emp/uw_emp.h5

Exercises
=========

1) Compare numpy vs cupy performance
------------------------------------
A numpy array can be converted to a cupy array,
and then you operate on it as before.

Look inside
center_matrix_naive.py

and then try to execute it with
# Force the use if the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 python center_matrix_naive.py

2) Compare numpy vs cupy libraries
------------------------------------
A numpy array can be converted to a cupy array,
and then used with either library.

Look inside
lingalg_qr.py

and then try to execute it with
# Force the use if the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 python lingalg_qr.py


3) Compare scipy vs cupyx libraries
-----------------------------------
A numpy array can be converted to a cupy array,
and then used with scipy.
GPU compute can be forced by using the cupyx.scipy library.

Look inside
scipy_math.py

and then try to execute it with
# Force the use if the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 python scipy_math.py


4) Mixing CPU and GPU code
--------------------------
A numpy array can be converted to a cupy array,
but those cupy arrays are not accepted in many scipy functions.
In those cases, we must explicitly cast the buffers back to numpy.

Look inside
scipy_mixed.py

and then try to execute it with
# Force the use if the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 python scipy_mixed.py


5) Enable unified memory
------------------------
So far, cupy was moving memory around behind the scenes.
Let's enable the true shared memory setup.

Look inside
center_matrix_naive_apu.py
(and compare with center_matrix_naive.py)

and then try to execute it with
# Force the use if the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
# shared memory is disabled by default, must set the two env variables
CUPY_ENABLE_UMP=1 HSA_XNACK=1 taskset -c 24-47,120-143 python center_matrix_naive_apu.py

As further exercise, try to modify the other python files along the same lines, too.

