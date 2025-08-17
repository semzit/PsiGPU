#include "circuit_executer.h"
#include "gates.h"
#include "quantum_state.h"
#include "circuit.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <unordered_map>

const int THREADS_PER_BLOCK = 256; 

__global__ void applyGate(cuDoubleComplex* stateVec, const cuDoubleComplex* U, int numQubits); 

dim3 CircuitExecuter::calculateLaunchDims(int total_elements) const {
    int num_blocks = static_cast<int>(std::ceil(static_cast<double>(total_elements) / THREADS_PER_BLOCK)); 
    return dim3(num_blocks, THREADS_PER_BLOCK); 
}

void CircuitExecuter::execute(Circuit& circuit, QuantumState& state) {
    dim3 launch_dims = calculateLaunchDims(state.num_amplitudes); 
    unsigned int num_blocks = launch_dims.x; 
    unsigned int num_threads = launch_dims.y; 
    
    prepareMatrix(circuit); 

    cuDoubleComplex* d_matrix;
    size_t matSize = circuit.completeMatrix.size() * sizeof(cuDoubleComplex);
    cudaMalloc(&d_matrix, matSize);
    cudaMemcpy(d_matrix, circuit.completeMatrix.data(), matSize, cudaMemcpyHostToDevice);

    applyGate<<<num_blocks, num_threads>>>(state.amplitudes, d_matrix, circuit.qubit_count);

    cudaFree(d_matrix);

    cudaDeviceSynchronize(); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    }
}

// Only handles single-qubit gates
void CircuitExecuter::prepareMatrix(Circuit& circuit) {
    int numQubits = circuit.qubit_count;

    std::vector<std::vector<cuDoubleComplex>> qubitMatrices(numQubits, std::vector<cuDoubleComplex>{
        {1,0}, {0,0},
        {0,0}, {1,0}
    });
    
    for (const Gate& gate : circuit.gates) {
        if (gate.targets.size() != 1) {
            throw std::runtime_error("Only single-qubit gates supported in prepareMatrix.");
        }
        int q = gate.targets[0];
        qubitMatrices[q] = multiplyMatrices(gate.matrix, qubitMatrices[q], 2, 2);
    }

    std::vector<cuDoubleComplex> finalMatrix = qubitMatrices[0];
    for (int q = 1; q < numQubits; ++q) {
        finalMatrix = tensorProduct(finalMatrix, qubitMatrices[q], 1 << q, 2);
    }

    circuit.completeMatrix = finalMatrix;
}

bool overlaps(const Gate& a, const Gate& b) {
    for (int qa: a.targets) {
        for(int qb: b.targets) {
            if (qa == qb) return true;
        }
    }
    return false;
}

std::vector<cuDoubleComplex> tensorProduct(const std::vector<cuDoubleComplex>& A, const std::vector<cuDoubleComplex>& B, int dimA, int dimB) {
    std::vector<cuDoubleComplex> result(dimA * dimB * dimA * dimB);
    for (int i = 0; i < dimA; ++i) {
        for (int j = 0; j < dimA; ++j) {
            for (int k = 0; k < dimB; ++k) {
                for (int l = 0; l < dimB; ++l) {
                    result[(i*dimB + k) * (dimA*dimB) + (j*dimB + l)] = cuCmul(A[i*dimA + j], B[k*dimB + l]);
                }
            }
        }
    }
    return result;
}

std::vector<cuDoubleComplex> multiplyMatrices(const cuDoubleComplex* A, const std::vector<cuDoubleComplex>& B, int dimA, int dimB) {
    std::vector<cuDoubleComplex> result(dimA*dimA, make_cuDoubleComplex(0,0));
    for (int i = 0; i < dimA; ++i) {
        for (int j = 0; j < dimA; ++j) {
            cuDoubleComplex sum = make_cuDoubleComplex(0,0);
            for (int k = 0; k < dimA; ++k) {
                sum = cuCadd(sum, cuCmul(A[i*dimA + k], B[k*dimA + j]));
            }
            result[i*dimA + j] = sum;
        }
    }
    return result;
}

__global__ void applyGate(cuDoubleComplex* stateVec, const cuDoubleComplex* U, int numQubits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << numQubits;
    if (idx >= dim) return;

    extern __shared__ cuDoubleComplex temp[];
    int tIdx = threadIdx.x;
    temp[tIdx] = stateVec[idx];
    __syncthreads();


    cuDoubleComplex sum = make_cuDoubleComplex(0,0);
    int gateDim = 1 << numQubits;
    for(int j = 0; j < gateDim; ++j) {
        sum = cuCadd(sum, cuCmul(U[idx * gateDim + j], temp[j]));
    }

    stateVec[idx] = sum;
}
