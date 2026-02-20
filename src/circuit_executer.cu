#include "circuit_executer.h"
#include "gates.h"
#include "quantum_state.h"
#include "circuit.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <unordered_map>

const int THREADS_PER_BLOCK = 256; 

__global__ void applyGate(cuDoubleComplex* stateVec,const Gate* gates, int numQubits, int numGates); 

/**
 * Execute the circuit
 */
void CircuitExecuter::execute(Circuit& circuit, QuantumState& state) {
    int num_blocks = (state.num_amplitudes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // allocate a cuDoubleComplex the size 4 * numGates * sizeof(cuDoubleComplex)
    Gate* d_gates;
    size_t matSize = circuit.gates.size() * sizeof(Gate); 
    cudaMalloc(&d_gates, matSize);
    cudaMemcpy(d_gates, circuit.gates.data(), matSize, cudaMemcpyHostToDevice);

    applyGate<<<num_blocks, THREADS_PER_BLOCK>>>(state.amplitudes, d_gates, circuit.qubit_count, static_cast<int>(circuit.gates.size()));

    cudaFree(d_gates);

    cudaDeviceSynchronize(); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    }
}

/**
 * Cuda Kernal for applying the gate 
 * iterates through the gates
 */
__global__ void applyGate(cuDoubleComplex* stateVec,const Gate* gates, int numQubits, int numGates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    int dim = 1 << numQubits;  
    if (idx >= dim) return;

    cuDoubleComplex amp = stateVec[idx];  

    for (int gate = 0; gate < numGates; gate++){
        int i = gates[gate].targets[0]; 
        int pairIdx = idx ^ (1 << i); 

        if(idx < pairIdx){
            cuDoubleComplex a = stateVec[idx]; 
            cuDoubleComplex b = stateVec[pairIdx]; 

            stateVec[idx] =     cuCadd(cuCmul(gates[gate].matrix[0], a), 
                                       cuCmul(gates[gate].matrix[1], b)); 
            
            stateVec[pairIdx] = cuCadd(cuCmul(gates[gate].matrix[2], a), 
                                       cuCmul(gates[gate].matrix[3], b)); 
        }
        __syncthreads(); 
    }
}
