#include <cuComplex.h>
#include "quantum_state.h"

/**
 * iterate through every amplitude and calculate probability 
 */
__global__ void calculate_probability(cuDoubleComplex* state_vector, double* probabilities, int num_qubits){
    int  total_elements = 1ULL << num_qubits;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total_elements){
        cuDoubleComplex amplitude = state_vector[i];
        probabilities[i] = amplitude.x * amplitude.x + amplitude.y * amplitude.y;
    }
}

/**
 * iterate through every amplitude and set it to 0 instead of the greates one which is set to 1.0
 */ 
__global__ void measure(cuDoubleComplex* state_vector, double* probabilities, int num_qubits, int winning_index){
    int  total_elements = 1ULL << num_qubits;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total_elements && i != winning_index){
        state_vector[i] = make_cuDoubleComplex(0.0, 0.0);
    }else if (i < total_elements){
        state_vector[i] = make_cuDoubleComplex(1.0, 0.0);
    }
}

void measureAndCollapse(QuantumState& state){
    
} 
