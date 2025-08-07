#ifndef CIRCUITEXECUTER_H
#define CIRCUITEXECUTER_H
#include <cuComplex.h>

extern __global__ void cnot_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit, int control_qubit); 
extern __global__ void swap_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit, int control_qubit); 
extern __global__ void toffoli_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit, int control_qubit); 
extern __global__ void pauli_x_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit); 
extern __global__ void pauli_y_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit); 
extern __global__ void pauli_z_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit); 
extern __global__ void s_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit); 
extern __global__ void t_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit); 
extern __global__ void x_rotation_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit); 
extern __global__ void y_rotation_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit); 
extern __global__ void z_rotation_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit); 
extern __global__ void hadamard_kernal(cuDoubleComplex* state_vector, int num_qubits, int target_qubit); 

struct CircuitExecuter{
    public: 
        CircuitExecuter() = default; 
        void execute(const Circuit& circuit, QuantumState& state); 
    
    private: 
        // helper for determining the proper grid and block launch dimensions  
        dim3 calculateLaunchDims(int total_elements, int threads_per_block) const; 
};


#endif