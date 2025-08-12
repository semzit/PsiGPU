#include "../../include/circuit_executer.h"
#include "../../include/gates.h"
#include "../../include/quantum_state.h"
#include "../../include/circuit.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

const int THREADS_PER_BLOCK = 256; 


extern void measureAndCollapse(QuantumState& state); 

/**
 * calculate the kernel launch dim for a certain gate (num_blocks, THREADS_PER_BLOCK)
 * threads per block is GPU specific and usaully set at 256
 * numblocks is determined by dividing the total elements 2^num qubits by the threads per block
 */
dim3 CircuitExecuter::calculateLaunchDims(int total_elements)const{

    int num_blocks = static_cast<int>(std::ceil(static_cast<double>(total_elements)/ THREADS_PER_BLOCK)); 

    return dim3(num_blocks, THREADS_PER_BLOCK); 
}

/**
 * Iterates through the circuit and apply the gates to to state
 */
void CircuitExecuter::execute(const Circuit& circuit, QuantumState& state){
    dim3 launch_dims = calculateLaunchDims(state.num_amplitudes); 
    
    unsigned int num_blocks = launch_dims.x; 
    unsigned int num_threads = launch_dims.y; 

    for (Gate gate: circuit.gates){
        switch (gate.type){
            case GateType::Hadamard:{
                hadamard_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit); 
                break;
            }
            case GateType::PauliX:{
                pauli_x_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit); 
                break;
            }
            case GateType::PauliY:{
                pauli_y_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit); 
                break;
            }
            case GateType::PauliZ: {
                pauli_z_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit); 
                break;
            }
            case GateType::PhaseS: {
                s_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit); 
                break;
            }
            case GateType::PhaseT: {
                t_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit); 
                break;
            }
            case GateType::RotationX:{
                x_rotation_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit); 
                break;
            }
            case GateType::RotationY:{
                y_rotation_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit); 
                break;
            }
            case GateType::RotationZ:{
                z_rotation_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit); 
                break;
            }
            case GateType::CNOT:{
                cnot_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit, gate.control_qubit); 
                break;
            }
            case GateType::SWAP:{
                swap_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit, gate.control_qubit); 
                break;
            }
            case GateType::Toffoli: {
               toffoli_kernel<<<num_blocks, num_threads>>>(state.amplitudes, circuit.qubit_count, gate.target_qubit, gate.control_qubit, gate.control_qubit_two); 
               break;
            }
            default:{
                std::cerr << "Error: Unknown gate type encountered.\n";
                
                throw std::runtime_error("Unsupported gate type.");
            }
        }
        cudaDeviceSynchronize(); 
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after " << gate.toString() << " kernel launch: " << cudaGetErrorString(err) << std::endl;
        }
    }
    measureAndCollapse(state);
}

