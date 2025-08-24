#ifndef CIRCUIT_EXECUTER_H
#define CIRCUIT_EXECUTER_H
#include <cuComplex.h>
#include "quantum_state.h"
#include "circuit.h"

std::vector<cuDoubleComplex> tensorProduct(const std::vector<cuDoubleComplex>& A, const std::vector<cuDoubleComplex>& B, int dimA, int dimB);  
std::vector<cuDoubleComplex> multiplyMatrices(const cuDoubleComplex* A, const std::vector<cuDoubleComplex>& B, int dimA, int dimB); 

class CircuitExecuter{
    public: 
        CircuitExecuter() = default; 
        void execute(Circuit& circuit, QuantumState& state); 
        void prepareMatrix(Circuit& circuit) ; 
    private: 
        // helper for determining the proper grid and block launch dimensions  
        std::pair<dim3, dim3> calculateLaunchDims(int total_elements) const; 
};





#endif