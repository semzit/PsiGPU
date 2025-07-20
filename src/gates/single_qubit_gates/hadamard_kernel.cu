

__global__ void hadamard_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit){
    
    size_t global_idx = global_idx = blockIdx.x * blockDim.x + threadIdx.x;
}