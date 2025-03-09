all: all_gcc

all_gcc: mt1_gcc mt1_vect_gcc mt1_vectx4_gcc mt1_prefetch_gcc

all_clang: mt1_clang mt1_vect_clang mt1_vectx4_clang mt1_prefetch_clang

all_amd_gpu: mt1_amd_gpu mt1_prefetch16_amd_gpu mt1_prefetch1_amd_gpu

all_nv_gpu: mt1_nv_gpu mt1_prefetch16_nv_gpu mt1_prefetch1_nv_gpu

all_nv_cpu: mt1_nv_cpu mt1_prefetch16_nv_cpu

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

mt1_nv_gpu: mt1.cpp
	$(NV_CXX) -o $@ $<

mt1_nv_cpu: mt1.cpp
	$(NV_CXX_CPU) -o $@ $<


mt1_vect_gcc: mt1_vect.cpp
	# ideal vector width may be CPU specific (e.g. AVX512 vs SSE)
	$(GCC_CXX) -DVECT=16 -o $@ $<

mt1_vectx4_gcc: mt1_vect.cpp
	# ideal vector width may be CPU specific (e.g. AVX512 vs SSE)
	$(GCC_CXX) -DVECT=64 -o $@ $<

mt1_vect_clang: mt1_vect.cpp
	# ideal vector width may be CPU specific (e.g. AVX512 vs SSE)
	$(CLANG_CXX) -DVECT=16 -o $@ $<

mt1_vectx4_clang: mt1_vect.cpp
	# ideal vector width may be CPU specific (e.g. AVX512 vs SSE)
	$(CLANG_CXX) -DVECT=64 -o $@ $<



mt1_prefetch_gcc: mt1_vect.cpp
	$(GCC_CXX) -DVECT=16 -DDO_PREFETCH -o $@ $<

mt1_prefetchx4_gcc: mt1_vect.cpp
	$(GCC_CXX) -DVECT=64 -DDO_PREFETCH -o $@ $<

mt1_prefetch_clang: mt1_vect.cpp
	$(CLANG_CXX) -DVECT=16 -DDO_PREFETCH -o $@ $<

mt1_prefetch_amd_gpu: mt1_prefetch.cpp
	# There is no prefetch on GPUs, just keep the same as for the CPUs
	$(AMD_CXX) -DPREFETCH_DEPTH=16 -o $@ $<

mt1_prefetch_nv_gpu: mt1_prefetch.cpp
	# There is no prefetch on GPUs, just keep the same as for the CPUs
	$(NV_CXX) -DPREFETCH_DEPTH=16 -o $@ $<

mt1_prefetch_nv_cpu: mt1_prefetch.cpp
	$(NV_CXX_CPU) -DPREFETCH_DEPTH=16 -DPREFETCH_ASM_x86 -o $@ $<


clean:
	rm -f mt1_gcc mt1_clang mt1_amd_gpu mt1_nv_gpu mt1_nv_cpu 
	rm -f mt1_vect_gcc mt1_vect_clang mt1_vect_amd_gpu mt1_vect_nv_gpu mt1_vect_nv_cpu 
	rm -f mt1_vectx4_gcc mt1_vectx4_clang mt1_vectx4_nv_cpu 
	rm -f mt1_prefetch_gcc mt1_prefetch_clang mt1_prefetch_amd_gpu mt1_prefetch_nv_gpu mt1_prefetch_nv_cpu 

