all: all_gcc

all_gcc: cmp1_gcc cmp1_double_gcc cmp1_vect_gcc cmp1_vectx4_gcc cmp1_vect_double_gcc cmp1_vectx4_double_gcc cmp1_split_gcc cmp1_splitx4_gcc cmp1_split_double_gcc cmp1_splitx4_double_gcc cmp1_uni_gcc cmp1_uni_double_gcc

all_clang: cmp1_clang cmp1_double_clang cmp1_vect_clang cmp1_vectx4_clang cmp1_vect_double_clang cmp1_vectx4_double_clang cmp1_split_clang cmp1_splitx4_clang cmp1_split_double_clang cmp1_splitx4_double_clang cmp1_uni_clang cmp1_uni_double_clang

all_amd_gpu: cmp1_amd_gpu cmp1_double_amd_gpu cmp1_vectx4_amd_gpu cmp1_vectx4_double_amd_gpu cmp1_splitx4_amd_gpu cmp1_splitx4_double_amd_gpu cmp1_uni_amd_gpu cmp1_uni_double_amd_gpu

all_nv_gpu: cmp1_nv_gpu cmp1_double_nv_gpu cmp1_splitx4_nv_gpu cmp1_splitx4_double_nv_gpu cmp1_uni_nv_gpu cmp1_uni_double_nv_gpu

all_nv_cpu: cmp1_nv_cpu cmp1_double_nv_cpu cmp1_splitx4_nv_cpu cmp1_splitx4_double_nv_cpu cmp1_uni_nv_cpu cmp1_uni_double_nv_cpu

GCC_CXX   := g++ --std=c++17 -fopenmp -march=native -mtune=native -O3
CLANG_CXX := clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3
AMD_CXX   := amdclang++ --std=c++17 -march=native -mtune=native -O3 -fopenmp -DOMPGPU --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx90a,gfx942
NV_CXX   := nvc++ --std=c++17 -tp=native -O3 -mp -mp=gpu -DOMPGPU -gpu=ccall
NV_CXX_CPU := nvc++ --std=c++17 -tp=native -O3 -mp

cmp1_gcc: cmp1.cpp
	$(GCC_CXX) -o $@ $<

cmp1_double_gcc: cmp1.cpp
	$(GCC_CXX) -DTFLOAT=double -o $@ $<

cmp1_clang: cmp1.cpp
	$(CLANG_CXX) -o $@ $<

cmp1_double_clang: cmp1.cpp
	$(CLANG_CXX) -DTFLOAT=double -o $@ $<

cmp1_amd_gpu: cmp1.cpp
	$(AMD_CXX) -o $@ $<

cmp1_double_amd_gpu: cmp1.cpp
	$(AMD_CXX) -DTFLOAT=double -o $@ $<

cmp1_nv_gpu: cmp1.cpp
	$(NV_CXX) -o $@ $<

cmp1_double_nv_gpu: cmp1.cpp
	$(NV_CXX) -DTFLOAT=double -o $@ $<

cmp1_nv_cpu: cmp1.cpp
	$(NV_CXX_CPU) -o $@ $<

cmp1_double_nv_cpu: cmp1.cpp
	$(NV_CXX_CPU) -DTFLOAT=double -o $@ $<



cmp1_vect_gcc: cmp1_vect.cpp
	$(GCC_CXX) -o $@ $<

cmp1_vectx4_gcc: cmp1_vect.cpp
	$(GCC_CXX) -DVECT=64 -o $@ $<

cmp1_vect_double_gcc: cmp1_vect.cpp
	$(GCC_CXX) -DTFLOAT=double -o $@ $<

cmp1_vectx4_double_gcc: cmp1_vect.cpp
	$(GCC_CXX) -DTFLOAT=double -DVECT=32 -o $@ $<

cmp1_vect_clang: cmp1_vect.cpp
	$(CLANG_CXX) -o $@ $<

cmp1_vectx4_clang: cmp1_vect.cpp
	$(CLANG_CXX) -DVECT=64 -o $@ $<

cmp1_vect_double_clang: cmp1_vect.cpp
	$(CLANG_CXX) -DTFLOAT=double -o $@ $<

cmp1_vectx4_double_clang: cmp1_vect.cpp
	$(CLANG_CXX) -DTFLOAT=double -DVECT=32 -o $@ $<

cmp1_vectx4_amd_gpu: cmp1_vect.cpp
	$(AMD_CXX) -DVECT=4 -o $@ $<

cmp1_vectx4_double_amd_gpu: cmp1_vect.cpp
	$(AMD_CXX) -DTFLOAT=double -DVECT=4 -o $@ $<

# No NV vect versions, compiler does not support it in GPU code (as of 25.1)
#cmp1_vectx4_nv_gpu: cmp1_vect.cpp
#	$(NV_CXX) -DVECT=4 -o $@ $<




cmp1_split_gcc: cmp1_split.cpp
	$(GCC_CXX) -o $@ $<

cmp1_splitx4_gcc: cmp1_split.cpp
	$(GCC_CXX) -DVECT=64 -o $@ $<

cmp1_split_double_gcc: cmp1_split.cpp
	$(GCC_CXX) -DTFLOAT=double -o $@ $<

cmp1_splitx4_double_gcc: cmp1_split.cpp
	$(GCC_CXX) -DTFLOAT=double -DVECT=32 -o $@ $<

cmp1_split_clang: cmp1_split.cpp
	$(CLANG_CXX) -o $@ $<

cmp1_splitx4_clang: cmp1_split.cpp
	$(CLANG_CXX) -DVECT=64 -o $@ $<

cmp1_split_double_clang: cmp1_split.cpp
	$(CLANG_CXX) -DTFLOAT=double -o $@ $<

cmp1_splitx4_double_clang: cmp1_split.cpp
	$(CLANG_CXX) -DTFLOAT=double -DVECT=32 -o $@ $<

cmp1_split_nv_cpu: cmp1_split.cpp
	$(NV_CXX_CPU) -o $@ $<

cmp1_splitx4_nv_cpu: cmp1_split.cpp
	$(NV_CXX_CPU) -DVECT=64 -o $@ $<

cmp1_split_double_nv_cpu: cmp1_split.cpp
	$(NV_CXX_CPU) -DTFLOAT=double -o $@ $<

cmp1_splitx4_double_nv_cpu: cmp1_split.cpp
	$(NV_CXX_CPU) -DTFLOAT=double -DVECT=32 -o $@ $<

cmp1_splitx4_amd_gpu: cmp1_split.cpp
	$(AMD_CXX) -DVECT=4 -o $@ $<

cmp1_splitx4_double_amd_gpu: cmp1_split.cpp
	$(AMD_CXX) -DTFLOAT=double -DVECT=4 -o $@ $<

cmp1_splitx4_nv_gpu: cmp1_split.cpp
	$(NV_CXX) -DVECT=4 -o $@ $<

cmp1_splitx4_double_nv_gpu: cmp1_split.cpp
	$(NV_CXX) -DTFLOAT=double -DVECT=4 -o $@ $<


cmp1_uni_gcc: cmp1_uni.cpp
	# gcc does well with vector registers
	$(GCC_CXX) -DVECT=16 -o $@ $<

cmp1_uni_double_gcc: cmp1_uni.cpp
	# gcc does well with vector registers
	$(GCC_CXX) -DTFLOAT=double -DVECT=8 -o $@ $<

cmp1_uni_clang: cmp1_uni.cpp
	# clang does well with vector registers
	$(CLANG_CXX) -DVECT=16 -o $@ $<

cmp1_uni_double_clang: cmp1_uni.cpp
	# clang does well with vector registers
	$(CLANG_CXX) -DTFLOAT=double -DVECT=8 -o $@ $<

cmp1_uni_amd_gpu: cmp1_uni.cpp
	# clang seems to like vector notation even for GPUs
	$(AMD_CXX) -DVECT=8 -o $@ $<

cmp1_uni_double_amd_gpu: cmp1_uni.cpp
	# GPU code does not need explicit vectors
	$(AMD_CXX) -DTFLOAT=double -DVECT=1 -o $@ $<

cmp1_uni_nv_gpu: cmp1_uni.cpp
	# NVIDIA compiler does not support vectors in GPU code
	$(NV_CXX) -DVECT=1 -o $@ $<

cmp1_uni_double_nv_gpu: cmp1_uni.cpp
	# NVIDIA compiler does not support vectors in GPU code
	$(NV_CXX) -DTFLOAT=double -DVECT=1 -o $@ $<

cmp1_uni_nv_cpu: cmp1_uni.cpp
	# NVIDIA compiler does well with vector registers on CPU code
	$(NV_CXX_CPU) -DVECT=16 -o $@ $<

cmp1_uni_double_nv_cpu: cmp1_uni.cpp
	# NVIDIA compiler does well with vector registers on CPU code
	$(NV_CXX_CPU) -DTFLOAT=double -DVECT=8 -o $@ $<




clean:
	rm -f cmp1_gcc cmp1_clang cmp1_amd_gpu cmp1_nv_gpu cmp1_nv_cpu 
	rm -f cmp1_double_gcc cmp1_double_clang cmp1_double_amd_gpu cmp1_double_nv_gpu cmp1_double_nv_cpu
	rm -f cmp1_vect_gcc cmp1_vect_clang
	rm -f cmp1_vect_double_gcc cmp1_vect_double_clang
	rm -f cmp1_vectx4_gcc cmp1_vectx4_clang cmp1_vectx4_amd_gpu
	rm -f cmp1_vectx4_double_gcc cmp1_vectx4_double_clang cmp1_vectx4_double_amd_gpu
	rm -f cmp1_split_gcc cmp1_split_clang
	rm -f cmp1_split_double_gcc cmp1_split_double_clang
	rm -f cmp1_splitx4_gcc cmp1_splitx4_clang cmp1_splitx4_amd_gpu cmp1_splitx4_nv_gpu cmp1_splitx4_nv_cpu
	rm -f cmp1_splitx4_double_gcc cmp1_splitx4_double_clang cmp1_splitx4_double_amd_gpu cmp1_splitx4_double_nv_gpu cmp1_splitx4_double_nv_cpu
	rm -f cmp1_uni_gcc cmp1_uni_clang cmp1_uni_amd_gpu cmp1_uni_nv_gpu cmp1_uni_nv_cpu
	rm -f cmp1_uni_double_gcc cmp1_uni_double_clang cmp1_uni_double_amd_gpu cmp1_uni_double_nv_gpu cmp1_uni_double_nv_cpu

