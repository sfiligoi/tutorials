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

clean:
	rm -f dynamic_memory triad_cpu triad_gpu center_matrix_cpu center_matrix_gpu
