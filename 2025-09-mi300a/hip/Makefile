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

permanova_func_omp.o: permanova_func_omp.cpp
	$(CXX) $(GPUFLAGS) -DOMPGPU -o permanova_func_omp.o -c permanova_func_omp.cpp

permanova_func_hip.o: permanova_func_hip.hip
	$(HIPCXX) $(GPUFLAGS) -o permanova_func_hip.o -c permanova_func_hip.hip

permanova: permanova_func_omp.o permanova_func_hip.o permanova.cpp
	$(CXX) $(GPUFLAGS) $(HIPLIBFLAGS) -o permanova permanova.cpp permanova_func_omp.o permanova_func_hip.o -lamdhip64

wmma:
	$(CXX) $(GPUFLAGS) $(HIPLIBFLAGS) -DNDEBUG -o wmma_simple_hgemm -x hip wmma_simple_hgemm.cpp 
	$(CXX) $(GPUFLAGS) $(HIPLIBFLAGS) -DNDEBUG -o wmma_simple_sgemm -x hip wmma_simple_sgemm.cpp 
	$(CXX) $(GPUFLAGS) $(HIPLIBFLAGS) -DNDEBUG -o wmma_perf_hgemm -x hip wmma_perf_hgemm.cpp 
	$(CXX) $(GPUFLAGS) $(HIPLIBFLAGS) -DNDEBUG -o wmma_perf_sgemm -x hip wmma_perf_sgemm.cpp 

clean:
	rm -f *.o
	rm -f triad_omp triad_hip triad_hip_opt permanova
	rm -f simple_hgemm simple_sgemm perf_hgemm perf_sgemm

