CXX = amdclang++ --std=c++17 -O3 -fopenmp
GPUFLAGS = --offload-arch=gfx942:xnack+


all:
	# Must explicitly select one binary

dynamic_memory: dynamic_memory.cpp
	$(CXX) $(GPUFLAGS) -o dynamic_memory dynamic_memory.cpp

triad: triad_cpu triad_gpu
	# built triad_cpu triad_gpu

triad_cpu: triad.cpp
	$(CXX) -o triad_cpu triad.cpp

triad_gpu: triad.cpp
	$(CXX) $(GPUFLAGS) -DOMPGPU -o triad_gpu triad.cpp

center_matrix: center_matrix_cpu center_matrix_gpu
	# built center_matrix_cpu center_matrix_gpu

center_matrix_cpu: center_matrix_omp.cpp
	$(CXX) -o center_matrix_cpu center_matrix_omp.cpp

center_matrix_gpu: center_matrix_omp.cpp
	$(CXX) $(GPUFLAGS) -DOMPGPU -o center_matrix_gpu center_matrix_omp.cpp

partial_parallel: partial_parallel_cpu partial_parallel_gpu
	# built partial_parallel_cpu partial_parallel_gpu

partial_parallel_cpu: partial_parallel.cpp
	$(CXX) -o partial_parallel_cpu partial_parallel.cpp

partial_parallel_gpu: partial_parallel.cpp
	$(CXX) $(GPUFLAGS) -DOMPGPU -o partial_parallel_gpu partial_parallel.cpp

omp_ext: omp_ext_cpu omp_ext_gpu
	# built omp_ext_cpu omp_ext_gpu

omp_ext_cpu: omp_ext.cpp omp_ext_func.cpp
	$(CXX) -o omp_ext_func_cpu.o -c omp_ext_func.cpp
	$(CXX) -o omp_ext_cpu omp_ext.cpp omp_ext_func_cpu.o
	rm -f omp_ext_func_cpu.o

omp_ext_gpu: omp_ext.cpp omp_ext_func.cpp omp_ext_func_gpu_wrap.cpp
	$(CXX) $(GPUFLAGS) -DOMPGPU -o omp_ext_func_gpu.o -c omp_ext_func_gpu_wrap.cpp
	$(CXX) $(GPUFLAGS) -DOMPGPU -o omp_ext_gpu omp_ext.cpp omp_ext_func_gpu.o
	rm -f omp_ext_func_gpu.o

clean:
	rm -f dynamic_memory triad_cpu triad_gpu center_matrix_cpu center_matrix_gpu partial_parallel_cpu partial_parallel_gpu
	rm -f omp_ext_cpu omp_ext_gpu

