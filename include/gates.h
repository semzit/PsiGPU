#ifndef GATES_H
#define GATES_H

#include <math.h>
#include <vector>   
#include <string>   
#include <stdexcept> 
#include <cuComplex.h>

enum class GateType {
    PauliX, 
    PauliY, 
    PauliZ, 
    PhaseS, 
    PhaseT, 
    RotationX, 
    RotationY, 
    RotationZ, 
    Hadamard,
    CNOT,
    SWAP, 
    Toffoli
}; 

struct Gate{
    GateType type;
    int targets[3]; 
    const cuDoubleComplex* matrix; 
};

struct GateMatrices{
    static const cuDoubleComplex PauliX[4];
    static const cuDoubleComplex PauliY[4];
    static const cuDoubleComplex PauliZ[4];
    static const cuDoubleComplex PhaseS[4]; 
    static const cuDoubleComplex PhaseT[4]; 
    static const cuDoubleComplex RotationX[4]; 
    static const cuDoubleComplex RotationY[4]; 
    static const cuDoubleComplex RotationZ[4];
    static const cuDoubleComplex Hadamard[4];
    static const cuDoubleComplex CNOT[16]; 
    static const cuDoubleComplex SWAP[16]; 
    static const cuDoubleComplex Toffoli[64]; 
};

#endif