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
            
        }
        cudaDeviceSynchronize(); 
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after " << gate.toString() << " kernel launch: " << cudaGetErrorString(err) << std::endl;
        }
    }
    measureAndCollapse(state);
}

void CircuitExecuter::buildFusedMatrix(const Circuit& circuit){
    auto fused = std::vector<cuDoubleComplex>{
        make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), 
        make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0)
    }; 
    int fused_size = 2; 

    for (size_t i = 0; i< circuit.qubit_count; ++i){
        Gate* gateForQubit = nullptr;

        for (size_t j = 0; j < circuit.gates.size(); ++j) {
            for (size_t t = 0; t < circuit.gates[g].targets.size(); ++t) {
                if (circuit.gates[g].targets[t] == q) {
                    gateForQubit = &circuit.gates[g];
                    break;
                }
            }
            if (gateForQubit != nullptr) break;
        }

        if (gateForQubit != nullptr) {
            int gate_dim = (int)std::sqrt(gateForQubit->matrix.size());
            fused = tensorProduct(fused, fused_size, gateForQubit->matrix, gate_dim);
            fused_size *= gate_dim;
        } else {
            std::vector<cuDoubleComplex> I = {
                make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0),
                make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0)
            };
            fused = tensorProduct(fused, fused_size, I, 2);
            fused_size *= 2;
        }
    }
    timeSteps.push_back(fused); 
}

__global__ void applyFusedMatrix(cuDoubleComplex* stateVec,
                                 const cuDoubleComplex* U,
                                 int dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;

    extern __shared__ cuDoubleComplex temp[];
    temp[row] = stateVec[row];
    __syncthreads();

    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
    for (int col = 0; col < dim; ++col) {
        sum = cuCadd(sum, cuCmul(U[row * dim + col], temp[col]));
    }
    stateVec[row] = sum;
}

