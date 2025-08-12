#ifndef QUANTUM_STATE_H
#define QUANTUM_STATE_H

#include <cuda_runtime.h> 
#include <cuComplex.h> 
#include <vector>

// struct to hold the quantum state (amplitudes)
class QuantumState 
{
public: 
    cuDoubleComplex* amplitudes; 
    size_t num_qubits;           
    size_t num_amplitudes;       

    /**
     * constructor for quantum state allocates space on gpu for a list of amplitudes related to the number of qubits 
     */
    QuantumState(int n_qubits); 

    // destructor 
    ~QuantumState();
    
    void copyToHost(cuDoubleComplex *h_amplitudes) const;   
    void copyToDevice(cuDoubleComplex *h_amplitudes) const; 
    std::vector<cuDoubleComplex> getStateVector() const; 
}; 
#endif