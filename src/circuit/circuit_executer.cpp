#include <circuitExecuter.h>
#include <gates.h>

/**
 * calculate the kernel launch dim for a certain gate
 */
dim3 CircuitExecuter::calculateLaunchDims(int total_elements, int threads_per_block)const{

}

/**
 * should go through the circuit and apply the gates to to state
 */
void CircuitExecuter::execute(const Circuit& circuit, QuantumState& state){
    for (Gate gate: circuit.gates){
        dim3 launch_dims = calculateLaunchDims(); 
        switch (gate.type){
            case GateType::Hadamard:
            case GateType::PauliX: 
            case GateType::PauliY:
            case GateType::PauliZ: 
            case GateType::PhaseS: 
            case GateType::PhaseT: 
            case GateType::RotationX:
            case GateType::RotationY:
            case GateType::RotationZ:
                
            
            case GateType::CNOT: cnot_kernel(state.applitudes, circuit.qubit_count, gate.target_qubit, gate.control_qubit); 
            case GateType::SWAP: swap_kernel(state.applitudes, circuit.qubit_count, gate.target_qubit, gate.control_qubit); 
            case GateType::Toffoli:  toffoli_kernel(state.applitudes, circuit.qubit_count, gate.target_qubit, gate.control_qubit); 
        }
    }
}