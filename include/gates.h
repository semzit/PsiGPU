#ifndef GATES_H
#define GATES_H

#include <vector>   
#include <string>   
#include <stdexcept> 

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
    int target_qubit; 
    int control_qubit;
    int control_qubit_two; 
    std::vector<cuDoubleComplex> matrix; 
    
    Gate(GateType t, int target)
        : type(t), target_qubit(target), control_qubit(-1), control_qubit_two(-1){}

    Gate(GateType t, int target, int control)
        : type(t), target_qubit(target), control_qubit(control), control_qubit_two(-1){}

    Gate(GateType t, int target, int control, int control_two)
        : type(t), target_qubit(target), control_qubit(control), control_qubit_two(control_two){}

    std::string toString() const{
        std::string g; 
        switch (type){
            case GateType::Hadamard: g = "H"; break;
            case GateType::PauliX: g = "X"; break;
            case GateType::PauliY: g = "Y"; break; 
            case GateType::PauliZ: g = "Z"; break;
            case GateType::PhaseS: g = "S"; break;
            case GateType::PhaseT: g = "T"; break;
            case GateType::RotationX: g = "R_X"; break;
            case GateType::RotationY: g = "R_Y"; break;
            case GateType::RotationZ: g = "R_Z"; break;
            case GateType::CNOT: g = "CNOT"; break; 
            case GateType::SWAP: g = "SWAP"; break;
            case GateType::Toffoli: g = "Toffoli" ;
        }
        if (control_qubit == -1){
            g += "(Q" + std::to_string(target_qubit) + ")"; 
        }else if (control_qubit_two != -1){
            g += "(C" + std::to_string(control_qubit) + ", T" + std::to_string(target_qubit) + ")" + ", X" + std::to_string(control_qubit_two) + ")";
        }else{
            g += "(C" + std::to_string(control_qubit) + ", T" + std::to_string(target_qubit) + ")";
        }
        return g; 
    }
};

#endif