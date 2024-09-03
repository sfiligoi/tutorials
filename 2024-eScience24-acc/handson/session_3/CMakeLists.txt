# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.16)

# Project can mix languages - must have C++ support
# Kokkos flags are only applied to C++ files
project(Tut CXX)

# We build kokkos as a subdirectory of our project
add_subdirectory(${Tut_SOURCE_DIR}/.. ${Tut_BINARY_DIR}/kokkos)

add_executable(loop_kokkos loop_kokkos.cpp)

# This is the only line required to set up all compiler/linker flags
target_link_libraries(loop_kokkos Kokkos::kokkos)

