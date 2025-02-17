all: all_gcc

all_gcc: cmp1_gcc cmp1_double_gcc cmp1_vect_gcc cmp1_vectx4_gcc cmp1_vect_double_gcc cmp1_vectx4_double_gcc cmp1_split_gcc cmp1_splitx4_gcc cmp1_split_double_gcc cmp1_splitx4_double_gcc

all_clang: cmp1_clang cmp1_double_clang cmp1_vect_clang cmp1_vectx4_clang cmp1_vect_double_clang cmp1_vectx4_double_clang cmp1_split_clang cmp1_splitx4_clang cmp1_split_double_clang cmp1_splitx4_double_clang

all_amd_gpu: cmp1_amd_gpu cmp1_double_amd_gpu cmp1_vectx4_amd_gpu cmp1_vectx4_double_amd_gpu cmp1_splitx4_amd_gpu cmp1_splitx4_double_amd_gpu


cmp1_gcc: cmp1.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -o $@ $<

cmp1_double_gcc: cmp1.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -DTFLOAT=double -O3 -o $@ $<

cmp1_clang: cmp1.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -o $@ $<

cmp1_double_clang: cmp1.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DTFLOAT=double -o $@ $<

cmp1_amd_gpu: cmp1.cpp
	amdclang++ --std=c++17 -march=native -mtune=native -O3 -fopenmp -DOMPGPU --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx90a,gfx942 -o $@ $<

cmp1_double_amd_gpu: cmp1.cpp
	amdclang++ --std=c++17 -march=native -mtune=native -O3 -fopenmp -DOMPGPU --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx90a,gfx942 -DTFLOAT=double -o $@ $<



cmp1_vect_gcc: cmp1_vect.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -o $@ $<

cmp1_vectx4_gcc: cmp1_vect.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DVECT=64 -o $@ $<

cmp1_vect_double_gcc: cmp1_vect.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DTFLOAT=double -o $@ $<

cmp1_vectx4_double_gcc: cmp1_vect.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DTFLOAT=double -DVECT=32 -o $@ $<

cmp1_vect_clang: cmp1_vect.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -o $@ $<

cmp1_vectx4_clang: cmp1_vect.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DVECT=64 -o $@ $<

cmp1_vect_double_clang: cmp1_vect.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DTFLOAT=double -o $@ $<

cmp1_vectx4_double_clang: cmp1_vect.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DTFLOAT=double -DVECT=32 -o $@ $<

cmp1_vectx4_amd_gpu: cmp1_vect.cpp
	amdclang++ --std=c++17 -march=native -mtune=native -O3 -fopenmp -DOMPGPU --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx90a,gfx942 -DVECT=4 -o $@ $<

cmp1_vectx4_double_amd_gpu: cmp1_vect.cpp
	amdclang++ --std=c++17 -march=native -mtune=native -O3 -fopenmp -DOMPGPU --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx90a,gfx942 -DTFLOAT=double -DVECT=4 -o $@ $<



cmp1_split_gcc: cmp1_split.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -o $@ $<

cmp1_splitx4_gcc: cmp1_split.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DVECT=64 -o $@ $<

cmp1_split_double_gcc: cmp1_split.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DTFLOAT=double -o $@ $<

cmp1_splitx4_double_gcc: cmp1_split.cpp
	g++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DTFLOAT=double -DVECT=32 -o $@ $<

cmp1_split_clang: cmp1_split.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -o $@ $<

cmp1_splitx4_clang: cmp1_split.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DVECT=64 -o $@ $<

cmp1_split_double_clang: cmp1_split.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DTFLOAT=double -o $@ $<

cmp1_splitx4_double_clang: cmp1_split.cpp
	clang++ --std=c++17 -fopenmp -march=native -mtune=native -O3 -DTFLOAT=double -DVECT=32 -o $@ $<

cmp1_splitx4_amd_gpu: cmp1_split.cpp
	amdclang++ --std=c++17 -march=native -mtune=native -O3 -fopenmp -DOMPGPU --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx90a,gfx942 -DVECT=4 -o $@ $<

cmp1_splitx4_double_amd_gpu: cmp1_split.cpp
	amdclang++ --std=c++17 -march=native -mtune=native -O3 -fopenmp -DOMPGPU --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx90a,gfx942 -DTFLOAT=double -DVECT=4 -o $@ $<


clean:
	rm -f cmp1_gcc cmp1_clang cmp1_amd_gpu
	rm -f cmp1_double_gcc cmp1_double_clang cmp1_double_amd_gpu
	rm -f cmp1_vect_gcc cmp1_vect_clang
	rm -f cmp1_vect_double_gcc cmp1_vect_double_clang
	rm -f cmp1_vectx4_gcc cmp1_vectx4_clang cmp1_vectx4_amd_gpu
	rm -f cmp1_vectx4_double_gcc cmp1_vectx4_double_clang cmp1_vectx4_double_amd_gpu
	rm -f cmp1_split_gcc cmp1_split_clang
	rm -f cmp1_split_double_gcc cmp1_split_double_clang
	rm -f cmp1_splitx4_gcc cmp1_splitx4_clang cmp1_splitx4_amd_gpu
	rm -f cmp1_splitx4_double_gcc cmp1_splitx4_double_clang cmp1_splitx4_double_amd_gpu
