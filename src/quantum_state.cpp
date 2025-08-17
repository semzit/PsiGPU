#include "quantum_state.h"

QuantumState::QuantumState(int n_qubits) : num_qubits(n_qubits) {
    num_amplitudes = 1ULL << num_qubits;   // total amplitudes is 2^# of qubits

    cudaMalloc(&amplitudes, num_amplitudes * sizeof(cuDoubleComplex));   // allocate memory for which is a cuDoubleComplex of size 2^# of qubits

    cuDoubleComplex* h_amplitudes = (cuDoubleComplex*)calloc(num_amplitudes, sizeof(cuDoubleComplex)); // allocate amplitudes and set to 0
    h_amplitudes[0] = make_cuDoubleComplex(1.0, 0.0); // set first pair of ampitudes to 1, 0 resulting in a 100% probability of state being |00...0>
    
    copyToDevice(h_amplitudes); 
    free(h_amplitudes); // free on cpu
}

// Destructor 
QuantumState::~QuantumState() {
    if (amplitudes) {
        cudaFree(amplitudes);
        amplitudes = nullptr;
    }
}

/**
 * Method to get data on cpu neccessary for measurment 
 */
void QuantumState::copyToHost(cuDoubleComplex  *h_amplitudes) const {
    cudaMemcpy(h_amplitudes, amplitudes, num_amplitudes * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}

/**
 * Method to get data onto device 
 */
void QuantumState::copyToDevice(cuDoubleComplex  *h_amplitudes) const{
    cudaMemcpy(amplitudes, h_amplitudes, num_amplitudes * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice); // copy amplitudes to gpu
}

std::vector<cuDoubleComplex> QuantumState::getStateVector() const{
    std::vector<cuDoubleComplex> host_data(num_amplitudes);
    copyToHost(host_data.data());  // copy ampltudes to host_data
    return host_data;
}