Using backend GPU detection in skbio
====================================

Setup
=====
The following instructions work on SDSC Cosmos system.
Adapt as needed, if running on a different setup

Assuming you are inside a conda environment,
install cupy and some support libraries:
----------------------------------------
conda create -n skbio-tutorial -c conda-forge gxx_linux-64 libcblas liblapacke blas-devel make wget scikit-bio
conda activate skbio-tutorial
# AMD-GPU enabled skbio backend not avaialble in conda, build from source
wget https://github.com/scikit-bio/scikit-bio-binaries/archive/refs/tags/v1.0.3.tar.gz
tar -xzf v1.0.3.tar.gz
export AMD_HIP=Y
(cd scikit-bio-binaries-1.0.3 &&  make all -j)

Fetch large DistanceMatrix file and classification, used in examples:
--------------------------------------------------------------------
wget http://uaf-10.t2.ucsd.edu/~sfiligoi/unifrac_inputs/emp/uw_emp.h5
wget http://uaf-10.t2.ucsd.edu/~sfiligoi/unifrac_inputs/emp/emp_qiime_mapping_release1.tsv

Exercises
=========

1) GPU auto-detected and automatically used
-------------------------------------------
The backend will use the GPU, if one is detected.
Users still use normal numpy arrays.

Look inside
skbio_permanova.py 

and then try to execute it with
# Force the use of the 2nd APU on the node
export ROCR_VISIBLE_DEVICES=1
taskset -c 24-47,120-143 python skbio_permanova.py 

2) Forecfully run on CPU
------------------------
If for some reason one does not want to use a GPU (e.g., broken)
scikit-bio allows to force CPU execution.
(or don't build the AMD-GPU support)

Execute the following command
# Force the use of the 2nd APU on the node
SKBB_USE_GPU=N taskset -c 24-47,120-143 python skbio_permanova.py

