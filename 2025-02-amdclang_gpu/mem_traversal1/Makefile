all: all_gcc

all_gcc: mt1_gcc 

all_clang: mt1_clang 

all_amd_gpu: mt1_amd_gpu 

all_nv_gpu: mt1_nv_gpu 

all_nv_cpu: mt1_nv_cpu 

GCC_CXX   := g++ --std=c++17 -fopenmp -march=native -mtune=native -O3
CLANG_CXX := clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3
AMD_CXX   := amdclang++ --std=c++17 -march=native -mtune=native -O3 -fopenmp -DOMPGPU --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx90a,gfx942
NV_CXX   := nvc++ --std=c++17 -tp=native -O3 -mp -mp=gpu -DOMPGPU -gpu=ccall
NV_CXX_CPU := nvc++ --std=c++17 -tp=native -O3 -mp

mt1_gcc: mt1.cpp
	$(GCC_CXX) -o $@ $<

mt1_clang: mt1.cpp
	$(CLANG_CXX) -o $@ $<

mt1_amd_gpu: mt1.cpp
	$(AMD_CXX) -o $@ $<

mt1__nv_gpu: mt1.cpp
	$(NV_CXX) -o $@ $<

mt1_nv_cpu: mt1.cpp
	$(NV_CXX_CPU) -o $@ $<



clean:
	rm -f mt1_gcc mt1_clang mt1_amd_gpu mt1_nv_gpu mt1_nv_cpu 

