#include "circuit_executer.h"
#include <cuComplex.h>

__global__ void pauli_x_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID
    size_t grid_size = gridDim.x * blockDim.x;          // Total threads in the grid

    // Calculate the 'stride' or bitmask for the target qubit.
    // This is 2^target_qubit.
    size_t stride = 1ULL << target_qubit;

    // The total number of amplitude pairs is 2^(num_qubits - 1).
    size_t num_pairs = 1ULL << (num_qubits - 1);

    // Loop through all relevant pairs of amplitudes using the grid-stride loop.
    for (size_t current_pair_idx = tid; current_pair_idx < num_pairs; current_pair_idx += grid_size) {
        // Calculate the base index for this pair.
        // This index corresponds to the amplitude where the target_qubit's bit is 0.
        size_t base_idx = (current_pair_idx / stride) * (stride * 2) + (current_pair_idx % stride);

        // idx0 is the index where the target_qubit's bit is 0.
        size_t idx0 = base_idx;
        // idx1 is the index where the target_qubit's bit is 1.
        size_t idx1 = base_idx | stride; // Efficiently sets the target_qubit bit to 1

        // Load the two complex amplitudes involved in the transformation.
        cuDoubleComplex amp0 = state_vector[idx0];
        cuDoubleComplex amp1 = state_vector[idx1];

        // Apply the Pauli-X transformation (swap amp0 and amp1).
        // This is a direct implementation of the matrix:
        // new_amp0 = 0*amp0 + 1*amp1 = amp1
        // new_amp1 = 1*amp0 + 0*amp1 = amp0
        state_vector[idx0] = amp1;
        state_vector[idx1] = amp0;
    }
}