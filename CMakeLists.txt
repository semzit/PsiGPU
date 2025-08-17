cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(PsiGPU LANGUAGES CXX CUDA)

add_library(PsiGPULib
    src/circuit.cpp
    src/gates.cpp
    src/circuit_executer.cu
    src/quantum_state.cpp
)

set_target_properties(PsiGPULib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(PsiGPULib PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(PsiGPU
    src/main.cpp
)

target_link_libraries(PsiGPU PRIVATE PsiGPULib)


# Enable testing and add test directory
enable_testing()
add_subdirectory(test)
