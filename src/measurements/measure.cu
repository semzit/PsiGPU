#include <cuComplex.h>

// Go through every amplitude and calculate probability
__global__ void calculate_probability(cuDoubleComplex* state_vector, double* probabilities, int num_qubits){
    int i = blockIdx.x*blockDim.x + threadIdx.x;

}

// go through every amplitude and set it to 0 instead of the greates one which is set to 1.0
__global__ void measure(cuDoubleComplex* state_vector, double* probabilities, int num_qubits){
    int i = blockIdx.x*blockDim.x + threadIdx.x;

}
