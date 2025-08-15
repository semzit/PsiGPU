#ifndef CIRCUIT_H
#define CIRCUIT_H

#include <vector>
#include "gates.h"

class Circuit
{

public:
    int qubit_count; 
    std::vector<std::vector<cuDoubleComplex>> timeSteps; 
    std::vector<Gate> gates;     
    Circuit(int num_qubits);
    ~Circuit();
    void printCircuitCLI() const; 
    // single qubit gates
    void addHadamard(int target_qubit); 
    void addPualiX(int target_qubit); 
    void addPauliY(int target_qubit); 
    void addPauliZ(int target_qubit); 
    void addPhaseS(int target_qubit); 
    void addPhaseT(int target_qubit);
    void addRotationX(int target_qubit); 
    void addRotationY(int target_qubit); 
    void addRotationZ(int target_qubit); 
    // multi qubit gates
    void addCNOT(int target_qubit, int control); 
    void addSWAP(int target_qubit, int control); 
    void addToffoli(int target_qubit, int control, int control_two); 
};

#endif


