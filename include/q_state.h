#ifndef QUANTUM_STATE_H
#define QUANTUM_STATE_H

#include <cuda_runtime.h> 
#include <cuComplex.h> 

// struct to hold the quantum state (amplitudes)
struct QuantumState {
    cuDoubleComplex* amplitudes; 
    size_t num_qubits;           
    size_t num_amplitudes;       

    /**
     * constructor for quantum state allocates space on gpu for a list of amplitudes related to the number of qubits 
     */
    QuantumState(size_t n_qubits) : num_qubits(n_qubits) {
        num_amplitudes = 1ULL << num_qubits;   // total amplitudes is 2^# of qubits

        CUDA_CHECK(cudaMalloc(&amplitudes, num_amplitudes * sizeof(cuDoubleComplex)));  

        cuDoubleComplex* h_amplitudes = (cuDoubleComplex*)calloc(num_amplitudes, sizeof(cuDoubleComplex));
        if (h_amplitudes == NULL) {
            fprintf(stderr, "memory allocation on host failed!\n");
            exit(EXIT_FAILURE);
        }
        h_amplitudes[0] = make_cuDoubleComplex(1.0, 0.0); // set first pair of ampitudes to  1<1|  0<0| 

        CUDA_CHECK(cudaMemcpy(amplitudes, h_amplitudes, num_amplitudes * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice)); // copy amplitudes to gpu

        free(h_amplitudes); // free on cpu
    }

    // destructor 
    ~QuantumState() {
        if (amplitudes) {
            CUDA_CHECK(cudaFree(amplitudes));
            amplitudes = nullptr;
        }
    }

    // Dont allow copies
    QuantumState(const QuantumState&) = delete;
    QuantumState& operator=(const QuantumState&) = delete;

    /**
     * Method to get data on cpu neccessary for measurment 
     */
    void copyToHost(cuDoubleComplex* h_amplitudes) const {
        CUDA_CHECK(cudaMemcpy(h_amplitudes, amplitudes, num_amplitudes * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }
};

#endif