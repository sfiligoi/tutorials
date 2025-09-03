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

clean:
	rm -f dynamic_memory triad_cpu triad_gpu
