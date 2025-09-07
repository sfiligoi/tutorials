CXX = amdclang++ --std=c++17 -O3 -fopenmp
HIPCXX = hipcc --std=c++17 -O3
GPUFLAGS = --offload-arch=gfx942:xnack+
HIPLIBFLAGS =  -D__HIP_PLATFORM_AMD__ -I /opt/rocm-6.3.0/include -L /opt/rocm-6.3.0/lib

all:
	# Must explicitly select one binary

triad: triad_omp triad_hip triad_hip_opt
	# built triad_omp triad_hip triad_hip_opt

triad_main.o: triad_ext.cpp
	$(CXX) $(GPUFLAGS) -DOMPGPU -o triad_main.o -c triad_ext.cpp

triad_omp: triad_func.hip triad_main.o
	# c++ compiler expects a .cpp extension, not .hip, but else it can use it
	ln -fs triad_func.hip triad_func.cpp
	$(CXX) $(GPUFLAGS) -DOMPGPU -o triad_func_omp.o -c triad_func.cpp
	$(CXX) $(GPUFLAGS) -DOMPGPU -o triad_omp triad_main.o triad_func_omp.o
	rm -f triad_func_omp.o
	rm -f triad_func.cpp

triad_hip: triad_func.hip triad_main.o
	$(HIPCXX) $(GPUFLAGS) -o triad_func_hip.o -c triad_func.hip
	$(CXX) $(GPUFLAGS) $(HIPLIBFLAGS) -o triad_hip triad_main.o triad_func_hip.o -lamdhip64
	rm -f triad_func_hip.o

triad_hip_opt: triad_func.hip triad_main.o
	$(HIPCXX) $(GPUFLAGS) -DHIP_OPT -o triad_func_hip_opt.o -c triad_func.hip
	$(CXX) $(GPUFLAGS) $(HIPLIBFLAGS) -o triad_hip_opt triad_main.o triad_func_hip_opt.o -lamdhip64
	rm -f triad_func_hip_opt.o

clean:
	rm -f *.o
	rm -f triad_omp triad_hip triad_hip_opt

