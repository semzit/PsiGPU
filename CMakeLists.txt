cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Must set before project() so CUDA detection doesn't fail
set(CMAKE_CUDA_ARCHITECTURES 89)

project(PsiGPU LANGUAGES CXX CUDA)

# Add source files
add_executable(PsiGPU
    src/main.cpp
    src/circuit/circuit.cpp
    src/circuit/circuitExecuter.cpp
    src/gates/multi_qubit_gates/cnot_kernel.cu
    src/gates/multi_qubit_gates/swap_kernel.cu
    src/gates/multi_qubit_gates/toffoli_kernel.cu
    src/gates/single_qubit_gates/pauli_gates/pauli_x_kernel.cu
    src/gates/single_qubit_gates/pauli_gates/pauli_y_kernel.cu
    src/gates/single_qubit_gates/pauli_gates/pauli_z_kernel.cu
    src/gates/single_qubit_gates/phase_gates/s_kernel.cu
    src/gates/single_qubit_gates/phase_gates/t_kernel.cu
    src/gates/single_qubit_gates/rotation_gates/x_rotation_kernel.cu
    src/gates/single_qubit_gates/rotation_gates/y_rotation_kernel.cu
    src/gates/single_qubit_gates/rotation_gates/z_rotation_kernel.cu
    src/gates/single_qubit_gates/hadamard_kernel.cu
    src/measurements/calculate_probabilities_kernel.cu
    src/measurements/collapse_and_normalize_kernel.cu
    src/measurements/measure.cpp
    src/circuits.cpp
)

set_target_properties(PsiGPU PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 13
    CUDA_STANDARD_REQUIRED YES
)
