#include <cuComplex.h>

__global__ void hadamard_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit){
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x; 

    size_t stride = 1ULL << target_qubit; // calculates the second number in the pair because amplitudes are stored in a 2d array

    for (size_t current_pair_idx = global_idx; current_pair_idx < (1ULL << (num_qubits - 1)); current_pair_idx += grid_size){
        // Calculate the base index for this pair
        size_t base_idx = (current_pair_idx / stride) * (stride * 2) + (current_pair_idx % stride);

        size_t idx0 = base_idx;
        size_t idx1 = base_idx | stride;

        cuDoubleComplex amp0 = state_vector[idx0];
        cuDoubleComplex amp1 = state_vector[idx1];
   
        const double inv_sqrt2 = 1.0 / sqrt(2.0);

        state_vector[idx0] = make_cuDoubleComplex(cuCadd(amp0, amp1).x * inv_sqrt2, cuCadd(amp0, amp1).y * inv_sqrt2);
        state_vector[idx1] = make_cuDoubleComplex(cuCsub(amp0, amp1).x * inv_sqrt2, cuCsub(amp0, amp1).y * inv_sqrt2);
    }
}