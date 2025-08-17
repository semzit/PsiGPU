#include "gates.h"

const cuDoubleComplex GateMatrices::PauliX[4]{
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
    make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0)
}; 

const cuDoubleComplex GateMatrices::PauliY[4]{
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,-1), 
    make_cuDoubleComplex(0,1), make_cuDoubleComplex(0,0)
};

const cuDoubleComplex GateMatrices::PauliZ[4]{
    make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), 
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(-1,0)
};

const cuDoubleComplex GateMatrices::PhaseS[4]{
    make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), 
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,1)   // S = [[1,0],[0,i]]
};

const cuDoubleComplex GateMatrices::PhaseT[4]{
    make_cuDoubleComplex(1, 0),                    // top-left: 1
    make_cuDoubleComplex(0, 0),                    // top-right: 0
    make_cuDoubleComplex(0, 0),                    // bottom-left: 0
    make_cuDoubleComplex(cos(M_PI/4), sin(M_PI/4)) // bottom-right: e^(iπ/4)
};


const cuDoubleComplex GateMatrices::Hadamard[4]{
    make_cuDoubleComplex(1/sqrt(2),0), make_cuDoubleComplex(1/sqrt(2),0),
    make_cuDoubleComplex(1/sqrt(2),0), make_cuDoubleComplex(-1/sqrt(2),0)
};

const cuDoubleComplex GateMatrices::CNOT[16]{
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
};

const cuDoubleComplex GateMatrices::SWAP[16]{
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
};

const cuDoubleComplex GateMatrices::Toffoli[64]{
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), 
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0),
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0),
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0),
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0),
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0),
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0),
    make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0),
};