cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(PsiGPU LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(${CUDA_TOOLKIT_INCLUDE_DIRECTORY})



# Add source files
add_executable(quantum_sim
    src/main.cpp
    src/simulator.cu
    src/circuits.cpp
    include/gates.h
)

target_include_directories(PsiGPU PRIVATE include)

set_target_properties(PsiGPU PROPERTIES
    CUDA_ARCHITECTURES 86
)

target_compile_options(PsiGPU PRIVATE -Wall -Wextra)
